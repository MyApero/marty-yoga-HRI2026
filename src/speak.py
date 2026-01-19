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
# bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
VOICE_SPEED = 1.0
SAMPLE_RATE = 24000


class Speak:
    def __init__(
        self,
        move_marty_callback,
        analyze_ongoing_frame,
        generated_text_callback=lambda text: None,
        can_i_speak=lambda: True,
    ):
        self.move_marty_callback = move_marty_callback
        self.analyze_ongoing_frame = analyze_ongoing_frame
        self.generated_text_callback = generated_text_callback
        self.can_i_speak = can_i_speak

        # Queues
        self.request_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # State Management
        self.current_utterance_done_event = threading.Event()
        self.correction = None
        self.generated_text = ""

        # Load Models
        try:
            self.kokoro = Kokoro(lang_code="b")
            print("Kokoro loaded successfully.", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load Kokoro: {e}", file=sys.stderr)
            sys.exit(1)

        self.stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32"
        )
        
        self.active_tasks = 0
        self.lock = threading.Lock()

        # --- Initialize Threads ONCE ---
        # 1. TTS Worker
        threading.Thread(target=self._tts_worker, daemon=True).start()

        # 2. Player Worker
        threading.Thread(target=self._player_worker, daemon=True).start()

        # 3. Coordinator (Manages the flow)
        threading.Thread(target=self._coordinator_worker, daemon=True).start()

    def _tts_worker(self):
        """
        Consumes text_queue, generates audio, pushes to audio_queue.
        """
        while True:
            text = self.text_queue.get()
            if text is None:
                self.audio_queue.put(None)
                self.text_queue.task_done()
                continue  # Go back to waiting for next conversation
            try:
                generator = self.kokoro(text, voice=VOICE_NAME, speed=VOICE_SPEED)
                for _, (_, _, audio) in enumerate(generator):
                    if len(audio) > 0:
                        self.audio_queue.put(audio)
            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)
            self.text_queue.task_done()

    def _player_worker(self):
        """
        Consumes audio_queue, plays sound, triggers callbacks.
        """
        while True:
            first_chunk = self.audio_queue.get()
            # If we receive None immediately, it's an empty message, just signal done
            if first_chunk is None:
                self.current_utterance_done_event.set()
                self.audio_queue.task_done()
                continue

            self.stream.start()

            is_first_chunk = True
            while True:
                while not self.can_i_speak():
                    time.sleep(0.1)
                if is_first_chunk:
                    audio_chunk = first_chunk
                    is_first_chunk = False
                else:
                    audio_chunk = self.audio_queue.get()

                if audio_chunk is None:
                    self.audio_queue.task_done()
                    break

                if self.correction is not None:
                    current_keys = self.analyze_ongoing_frame().keys()
                    if not (self.correction.keys() <= current_keys):
                        # Drain the queue to stop playback
                        while not self.audio_queue.empty():
                            try:
                                self.audio_queue.get_nowait()
                                self.audio_queue.task_done()
                            except queue.Empty:
                                break
                        break

                self._play_chunk(self.stream, audio_chunk)
                self.audio_queue.task_done()

            self.stream.stop()
            self.current_utterance_done_event.set()

    def _play_chunk(self, stream, chunk):
        duration = len(chunk) / SAMPLE_RATE
        if self.move_marty_callback:
            self.move_marty_callback(duration)
        stream.write(chunk)

    def _coordinator_worker(self):
        """Manages the lifecycle of a request."""
        while True:
            request = self.request_queue.get()
            messages, model = request
            self.current_utterance_done_event.clear()

            self._run_ollama_generation(messages, model)
            self.text_queue.put(None)
            self.current_utterance_done_event.wait()
            with self.lock:
                self.active_tasks = max(0, self.active_tasks - 1)

            self.request_queue.task_done()

    def _run_ollama_generation(self, messages, model):
        """
        Generates text from Ollama and pushes sentences to text_queue.
        """
        try:
            stream = ollama.chat(
                model=model,
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

    def say(self, messages, model="llama3.1"):
        """
        Non-blocking call. Queues the message and returns immediately.
        """
        with self.lock:
            self.generated_text = ""
            self.active_tasks += 1
        self.request_queue.put((messages, model))

    def presentation(self):
        print("Marty is presenting!")

    def goodbye(self):
        print("Marty says goodbye!")

    def load_pose(self, pose):
        system_instruction = (
            "You are a friendly yoga coach. Introduce the pose to the student, briefly describing it while being encouraging. "
            "Keep it to 3 sentences max. with max sentence length of 20 words. "
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Pose details: {str(pose['description'])}"},
        ]
        self.say(messages)

    def corrective_feedback(self, correction: dict, pose):
        system_instruction = (
            "You are a yoga coach. Receive the corrective feedback. You're inside of a discussion, no mention similar to 'during this pose'. "
            "Keep it to 1 sentences max with max sentence length of 15 words. Be very concise, only useful words. "
            "Don't mention the numbers. No asterisks, No parentheses."
            "Speak in the present tense and address the student directly without his name. Be creative."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "system", "content": f"Pose details: {str(pose['description'])}"},
            {"role": "user", "content": str(correction)},
        ]
        self.say(messages, model="llama3.2")

    def end_pose_feedback(self, feedbacks):
        self.empty_queues()
        self.correction = None
        self.generated_text = ""
        system_instruction = (
            "You are a friendly yoga coach. Receive the analysis report. "
            "Keep it to 3 sentences max with max sentence length of 20 words. "
            "Highlight the weak points and suggest one improvement tip. Be encouraging and positive. "
            "No numbers, no asterisks and no parentheses."
            "You can use metaphors if needed. "
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Full feedback data: {str(feedbacks)}"},
        ]
        print("\n[Main Thread] Adding end-of-pose feedback to queue...")
        self.say(messages)

    def is_done(self):
        """
        Checks if there are any active tasks in the queues.
        """
        with self.lock:
            return self.active_tasks == 0

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
        self.current_utterance_done_event.set()
        with self.lock:
            self.active_tasks = 0
