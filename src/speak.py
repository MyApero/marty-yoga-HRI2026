import ollama
import sounddevice as sd

from kokoro import KPipeline as Kokoro
import threading
import queue
import re
import time
import sys

# --- Configuration ---
VOICE_NAME = "am_michael"
VOICE_LANG = "en-gb"
# bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
VOICE_SPEED = 1.0
SAMPLE_RATE = 24000
MARGIN_BEFORE_START_SECONDS = 0.0


class Speak:
    def __init__(
        self,
        move_marty_callback,
        analyze_ongoing_frame,
        generated_text_callback=lambda text: None,
        can_i_speak=lambda: True,
    ):
        self.move_marty_callback = move_marty_callback
        self.request_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        try:
            self.kokoro = Kokoro(lang_code="b")
            print("Kokoro loaded successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load Kokoro: {e}", file=sys.stderr)
            sys.exit(1)

        # --- Start Coordinator ---
        # This thread stays alive to manage sequential conversations
        self.coordinator_thread = threading.Thread(
            target=self._coordinator_worker, daemon=True
        )
        self.coordinator_thread.start()

        self.correction: dict = None
        self.analyze_ongoing_frame = analyze_ongoing_frame

        self.generated_text = ""
        self.generated_text_callback = generated_text_callback

        self.can_i_speak = can_i_speak

    def _tts_worker(self):
        """
        Consumes text_queue, generates audio, pushes to audio_queue.
        """
        while True:
            text = self.text_queue.get()
            if text is None:
                self.audio_queue.put(None)
                self.text_queue.task_done()
                break
            try:
                generator = self.kokoro(text, voice=VOICE_NAME, speed=VOICE_SPEED)
                for _, (_, _, audio) in enumerate(generator):
                    if len(audio) > 0:
                        self.audio_queue.put(audio)
            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)
            self.text_queue.task_done()

    def _player_worker(self, margin_before_start):
        """
        Consumes audio_queue, plays sound, triggers callbacks.
        """
        first_chunk = self.audio_queue.get()

        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        time.sleep(margin_before_start)
        while not self.can_i_speak():
            time.sleep(0.1)
        stream.start()

        chunk_index = 0

        if self.correction is not None:
            correction_keys = self.correction.keys()
            current_keys = self.analyze_ongoing_frame().keys()
            if not (correction_keys <= current_keys):
                # print(
                #     f"Cancelling: {correction_keys} not in {current_keys}",
                #     file=sys.stderr,
                # )
                self.audio_queue.task_done()
                self.empty_queues()
                stream.stop()
                stream.close()
                return

        while True:
            if chunk_index == 0:
                audio_chunk = first_chunk
            else:
                audio_chunk = self.audio_queue.get()

            if audio_chunk is None:
                self.audio_queue.task_done()
                break
            # Duration of the current audio chunk in seconds
            chunk_duration = len(audio_chunk) / SAMPLE_RATE
            if self.move_marty_callback:
                self.move_marty_callback(chunk_duration)

            stream.write(audio_chunk)
            self.audio_queue.task_done()
            chunk_index += 1

        stream.stop()
        stream.close()

    def say(self, messages, wait_before_first_chunk=MARGIN_BEFORE_START_SECONDS):
        """
        Non-blocking call. Queues the message and returns immediately.
        """
        self.generated_text = ""
        self.request_queue.put((messages, wait_before_first_chunk))

    def _coordinator_worker(self):
        """
        Main background loop. It processes one conversation request at a time
        to prevent audio overlapping.
        """
        while True:
            # Wait for a 'say' command
            request = self.request_queue.get()
            if request is None:
                break

            messages, wait_time = request

            # Start helper threads for THIS specific utterance
            tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
            player_thread = threading.Thread(
                target=self._player_worker, args=(wait_time,), daemon=True
            )

            tts_thread.start()
            player_thread.start()

            # Generate text (Producer)
            self._run_ollama_generation(messages)

            # Signal workers to finish
            self.text_queue.put(None)

            # Wait for the audio to fully finish before processing the next request
            tts_thread.join()
            player_thread.join()

            self.request_queue.task_done()

    def _run_ollama_generation(self, messages):
        """
        Generates text from Ollama and pushes sentences to text_queue.
        """
        try:
            stream = ollama.chat(
                model="llama3.2",
                messages=messages,
                stream=True,
            )

            buffer = ""
            sentence_endings = re.compile(r"(?<=[.!?])\s+")

            for chunk in stream:
                content = chunk["message"]["content"]
                # Print to stderr for debugging/logging
                # print(content, end="", file=sys.stderr, flush=True)

                buffer += content
                self.generated_text += content
                parts = sentence_endings.split(buffer)

                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            self.text_queue.put(sentence.strip())
                    buffer = parts[-1]

            self.generated_text_callback(self.generated_text)

            if buffer.strip():
                self.text_queue.put(buffer.strip())

        except Exception as e:
            print(f"Ollama Error: {e}", file=sys.stderr)

    # --- Wrapper methods for your main logic ---

    def presentation(self):
        print("Marty is presenting!")

    def goodbye(self):
        print("Marty says goodbye!")

    def corrective_feedback(self, correction: dict, pose):
        print("\n[Main Thread] Adding corrective feedback to queue...")
        system_instruction = (
            "You are a yoga coach. Receive the corrective feedback. "
            "Keep it to 1 sentences max. with max sentence length of 15 words. Be very concise and use only useful words. "
            "Don't mention the numbers. No asterisks, No parentheses."
            "Speak in the present tense and address the student directly without a name. Don't use 'throughout'."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "system", "content": f"Pose details: {str(pose['description'])}"},
            {"role": "user", "content": str(correction)},
        ]
        self.say(messages, wait_before_first_chunk=0)

    def end_pose_feedback(self, feedbacks):
        self.empty_queues()
        self.correction = None
        self.generated_text = ""
        system_instruction = (
            "You are a friendly yoga coach. Receive the analysis report. "
            "If Consistency > 80%, praise them. "
            "If Consistency < 50%, be encouraging but firm about the correction. "
            "Address the 'Primary Deviation' specifically. "
            "Keep it to 2 sentences max. with max sentence length of 20 words. "
            "Don't mention the numbers in the report. and don't put any asterisks and parentheses in the answer."
            "Be creative and don't hesitate to use metaphors and jokes! "
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Full feedback data: {str(feedbacks)}"},
        ]
        self.say(messages)

    def is_done(self):
        return (
            self.request_queue.empty()
            and self.text_queue.empty()
            and self.audio_queue.empty()
        )

    def wait_until_done(self):
        """
        Blocks until all queued messages have been processed.
        """
        self.request_queue.join()
        self.text_queue.join()
        self.audio_queue.join()

    def empty_queues(self):
        """
        Empties all queues immediately.
        """
        while self.request_queue.not_empty:
            try:
                self.request_queue.get_nowait()
                self.request_queue.task_done()
            except queue.Empty:
                break
        while self.text_queue.not_empty:
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except queue.Empty:
                break
        while self.audio_queue.not_empty:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
