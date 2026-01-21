import ollama
import sounddevice as sd
from kokoro import KPipeline as Kokoro
import threading
import queue
import re
import time
import sys
import soundfile as sf
import numpy as np

# --- Configuration ---
VOICE_NAME = "am_michael"
VOICE_SPEED = 1.0
SAMPLE_RATE = 24000


class Speak:
    def __init__(
        self,
        move_marty_callback,
        move_marty_enabled,
        move_marty_correctiv,
        analyze_ongoing_frame,
        generated_text_callback=lambda text: None,
        can_i_speak=lambda: True,
        audio_chunk_margin_seconds=3.0,
    ):
        self.move_marty_callback = move_marty_callback
        self.move_marty_enabled = move_marty_enabled
        self.move_marty_type_correctiv = None
        self.move_marty_correctiv = move_marty_correctiv
        self.analyze_ongoing_frame = analyze_ongoing_frame
        self.generated_text_callback = generated_text_callback
        self.can_i_speak = can_i_speak

        # Queues now store tuples: (payload, completion_event)
        self.request_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # State Management
        self.correction = None
        self.generated_text = ""
        self.active_tasks = 0
        self.lock = threading.Lock()

        self.audio_chunk_margin_seconds = audio_chunk_margin_seconds

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

        # --- Initialize Threads ---
        threading.Thread(target=self._tts_worker, daemon=True).start()
        threading.Thread(target=self._player_worker, daemon=True).start()
        threading.Thread(target=self._coordinator_worker, daemon=True).start()

    def _tts_worker(self):
        """
        Consumes text_queue, generates audio, pushes to audio_queue.
        Item format: (text_segment, completion_event)
        """
        while True:
            item = self.text_queue.get()
            text, event = item

            # If text is None, this signals the end of a specific request
            if text is None:
                self.audio_queue.put((None, event))
                self.text_queue.task_done()
                continue

            try:
                generator = self.kokoro(text, voice=VOICE_NAME, speed=VOICE_SPEED)
                for _, (_, _, audio) in enumerate(generator):
                    if len(audio) > 0:
                        # Push audio chunks with no event (event is only at the end)
                        self.audio_queue.put((audio, None))
            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)

            self.text_queue.task_done()

    def _player_worker(self):
        """
        Consumes audio_queue, plays sound.
        Item format: (audio_chunk, completion_event)
        """
        while True:
            item = self.audio_queue.get()
            audio_chunk, event = item

            # If we receive a marker with an event, it means the utterance finished
            if audio_chunk is None:
                if event:
                    event.set()  # Signal to the main thread that this specific sentence is done
                    with self.lock:
                        self.active_tasks = max(0, self.active_tasks - 1)
                self.audio_queue.task_done()
                continue

            # Start stream if not running (logic simplified here, stream can stay open or toggle)
            if not self.stream.active:
                self.stream.start()

            # Handle Interruption logic
            if self.correction is not None:
                current_keys = self.analyze_ongoing_frame().keys()
                # If correction is still valid, stop current playback
                if not (self.correction.keys() <= current_keys):
                    self._drain_queue_safely()
                    self.audio_queue.task_done()
                    self.move_marty_type_correctiv = None
                    continue

            # Wait if paused
            while not self.can_i_speak():
                time.sleep(0.1)

            self._play_chunk(self.stream, audio_chunk)
            self.audio_queue.task_done()

    def _drain_queue_safely(self):
        """
        Drains the audio queue. IMPORTANT: If we throw away an 'End of Request' marker,
        we MUST set its event, otherwise the main thread will wait forever.
        """
        while not self.audio_queue.empty():
            try:
                item = self.audio_queue.get_nowait()
                _, event = item
                if event:
                    event.set()  # Don't deadlock the waiter!
                    with self.lock:
                        self.active_tasks = max(0, self.active_tasks - 1)
                self.audio_queue.task_done()
            except queue.Empty:
                break
        # Stop stream if needed to cut off sound immediately
        # self.stream.stop()

    def _play_chunk(self, stream, chunk):
        duration = len(chunk) / SAMPLE_RATE
        if self.move_marty_enabled:
            if self.move_marty_callback and self.move_marty_type_correctiv is None:
                self.move_marty_callback(duration)
            elif self.move_marty_correctiv and self.move_marty_type_correctiv is not None:
                self.move_marty_correctiv(duration, self.move_marty_type_correctiv)
                self.move_marty_type_correctiv = None
        stream.write(chunk)

    def _coordinator_worker(self):
        """
        Consumes request_queue, runs Ollama.
        Item format: (messages, model, completion_event)
        """
        while True:
            request = self.request_queue.get()
            messages, model, event = request

            # 1. Generate text
            self._run_ollama_generation(messages, model)

            # 2. Push End-of-Request marker to TTS
            # The event travels: Coordinator -> TTS -> Player -> Event.set()
            self.text_queue.put((None, event))

            # 3. DO NOT WAIT. Immediately process next request.
            self.request_queue.task_done()

    def _run_ollama_generation(self, messages, model):
        try:
            stream = ollama.chat(model=model, messages=messages, stream=True)
            buffer = ""
            sentence_endings = re.compile(r"(?<=[.!?])\s+")

            for chunk in stream:
                content = chunk["message"]["content"]
                buffer += content
                self.generated_text += content
                parts = sentence_endings.split(buffer)

                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            # Push text with NO event
                            self.text_queue.put((sentence.strip(), None))
                    buffer = parts[-1]

            self.generated_text_callback(self.generated_text)
            if buffer.strip():
                self.text_queue.put((buffer.strip(), None))

        except Exception as e:
            print(f"Ollama Error: {e}", file=sys.stderr)

    # --- Public Methods ---

    def say(self, messages, model="llama3.1"):
        """
        Queues a message.
        Returns: threading.Event() that will be set when this specific message finishes playing.
        """
        completion_event = threading.Event()
        with self.lock:
            self.generated_text = ""
            self.active_tasks += 1

        self.request_queue.put((messages, model, completion_event))
        return completion_event

    def intro(self):
        system_instruction = (
            "You are a friendly yoga coach named Marty. Introduce yourself and greet the student warmly. "
            "Keep it to max length of 25 words. No -, use more dots than commas. "
            "Make it sound like a natural conversation."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": "Greet the student and introduce yourself as their yoga coach."},
        ]
        return self.say(messages, model="llama3.2")

    def start_counter(self):
        """
        Injects WAV directly into audio queue.
        Returns: threading.Event() that is set when audio finishes.
        """
        completion_event = threading.Event()
        file_path = "assets/countdown.wav"

        try:
            data, fs = sf.read(file_path, dtype="float32")
            if fs != SAMPLE_RATE:
                number_of_samples = round(len(data) * float(SAMPLE_RATE) / fs)
                data = np.interp(
                    np.linspace(0.0, 1.0, number_of_samples, endpoint=False),
                    np.linspace(0.0, 1.0, len(data)),
                    data,
                )
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            with self.lock:
                self.active_tasks += 1

            # Inject directly to audio queue
            # Format: (AudioData, None) -> (None, Event)
            self.audio_queue.put((data, None))
            self.audio_queue.put((None, completion_event))

        except FileNotFoundError:
            print(f"Error: {file_path} not found.", file=sys.stderr)
            # If failed, set event immediately so we don't block
            completion_event.set()
        except Exception as e:
            print(f"Error in start_counter: {e}", file=sys.stderr)
            completion_event.set()

        return completion_event

    def show_pose(self, pose):
        system_instruction = (
            "You are a friendly yoga coach. Explain the pose. "
            "Keep it simple in 2 sentences of 15 words"
            "Make it like you were in a discution"
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Pose details: {str(pose['description']['howto'])}"},
        ]
        return self.say(messages)

    def load_pose(self, pose):
        system_instruction = (
            "You are a friendly yoga coach. Introduce the pose, briefly describing it while being encouraging. "
            "Keep it to 2 sentences max with max sentence length of 15 words. "
            "At the end say you will demonstrate the pose"
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Pose details: {str(pose['description'])}"},
        ]
        return self.say(messages)

    def corrective_feedback(self, correction: dict, pose):
        # ... (rest of logic similar, just return self.say(...))
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
        return self.say(messages, model="llama3.2")

    def end_pose_feedback(self, feedbacks):
        self.empty_queues()  # This now handles setting events for cancelled items
        self.correction = None
        self.generated_text = ""
        system_instruction = (
            "You are a friendly yoga coach. Receive the analysis report. "
            "Keep it to 2 sentences max with max sentence length of 20 words. "
            "Highlight a weak points if needed and suggest one improvement tip but be encouraging and positive. "
            "No numbers, no asterisks and no parentheses."
            "You can use metaphors if needed. "
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Full feedback data: {str(feedbacks)}"},
        ]
        print("\n[Main Thread] Adding end-of-pose feedback to queue...")
        return self.say(messages)

    def is_done(self):
        with self.lock:
            return self.active_tasks == 0

    def wait_until_done(self):
        """Blocks until all tasks are done."""
        self.request_queue.join()
        self.text_queue.join()
        self.audio_queue.join()

    def empty_queues(self):
        """Empties queues and ENSURES events are set."""

        # Helper to drain a queue and set events
        def drain(q):
            while not q.empty():
                try:
                    item = q.get_nowait()
                    # Check structure based on which queue it is
                    # Audio/Text queue items are (payload, event)
                    # Request queue items are (msgs, model, event)
                    if isinstance(item, tuple) and len(item) >= 1:
                        event = item[-1]  # Event is always last
                        if isinstance(event, threading.Event):
                            event.set()
                    q.task_done()
                except queue.Empty:
                    break

        drain(self.request_queue)
        drain(self.text_queue)
        drain(self.audio_queue)

        with self.lock:
            self.active_tasks = 0
