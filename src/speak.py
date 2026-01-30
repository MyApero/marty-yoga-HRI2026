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
from src.extract_body_parts import extract_keywords, SYNONYMS

# --- Configuration ---
VOICE_NAME = "am_michael"
VOICE_SPEED = 1.0
SAMPLE_RATE = 24000
AUDIO_CHUNK_MARGIN_S = 6.0


COUNTDOWN_FILE_PATH = "assets/countdown.wav"
COUNTDOWN_SUBTITLES = "Get ready. 3... 2... 1... Hold!"

class Speak:
    def __init__(
        self,
        move_marty_callback,
        move_marty_enabled,
        move_marty_correctiv,
        analyze_ongoing_frame,
        can_i_speak=lambda: True,
        audio_chunk_margin_seconds=AUDIO_CHUNK_MARGIN_S,
    ):
        self.move_marty_callback = move_marty_callback
        self.move_marty_enabled = move_marty_enabled
        self.move_marty_type_correctiv = None
        self.move_marty_correctiv = move_marty_correctiv
        self.analyze_ongoing_frame = analyze_ongoing_frame
        self.can_i_speak = can_i_speak

        # Queues now store tuples with epoch: (epoch, payload, completion_event)
        self.request_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()

        # State Management
        self.correction = None
        self.generated_text = ""
        self.subtitles = ""
        self.active_tasks = 0
        self.lock = threading.Lock()

        self.memory = []

        self.current_epoch = 0
        self.current_sentence_keywords = set()

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

    #TODO TEST MEMORY
    def save_to_memory(self, text):
        if text == COUNTDOWN_SUBTITLES:
            return
        with self.lock:
            self.memory.append({"role": "assistant", "content": text})
            if len(self.memory) > 2: # Keep last 10 messages
                self.memory.pop(0)
            with open("voice_memory.toml", "w") as f:
                import toml

                toml.dump({"memory": self.memory}, f)

    def _tts_worker(self):
        """
        Consumes text_queue, generates audio, pushes to audio_queue.
        """
        while True:
            item = self.text_queue.get()
            epoch, text, event, generate_audio = item

            if epoch != self.current_epoch:
                if event:
                    event.set()
                self.text_queue.task_done()
                continue

            if text is None:
                self.audio_queue.put((epoch, None, None, event))
                self.text_queue.task_done()
                continue

            if not generate_audio:
                self.text_queue.task_done()
                continue

            try:
                generator = self.kokoro(text, voice=VOICE_NAME, speed=VOICE_SPEED)

                for i, (_, _, audio) in enumerate(generator):
                    if self.current_epoch != epoch:
                        print("TTS Aborted due to epoch change", file=sys.stderr)
                        if event:
                            event.set()
                        break

                    if len(audio) > 0:
                        chunk_subtitle = text if i == 0 else None
                        self.audio_queue.put((epoch, audio, chunk_subtitle, None))

                if self.current_epoch == epoch:
                    self.audio_queue.put(
                        (epoch, np.array([], dtype=np.float32), "", None)
                    )

            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)
                if event:
                    event.set()

            self.text_queue.task_done()

    def _player_worker(self):
        """
        Consumes audio_queue, plays sound.
        """
        buffer = []
        buffer_duration = 0.0
        current_playback_epoch = -1

        while True:
            item = self.audio_queue.get()
            epoch, audio_chunk, incoming_subtitles, event = item

            # 1. Stale Check
            if epoch != self.current_epoch:
                if event:
                    event.set()
                self.audio_queue.task_done()
                if current_playback_epoch == epoch:
                    buffer = []
                    buffer_duration = 0.0
                    self.subtitles = ""
                    self.current_sentence_keywords = set()
                continue

            current_playback_epoch = epoch

            # --- Case 1: End of Request Marker ---
            if audio_chunk is None:
                while len(buffer) > 0:
                    if self.current_epoch != epoch:
                        break
                    chunk_data, chunk_sub = buffer.pop(0)
                    if chunk_sub is not None:
                        self.subtitles = chunk_sub
                        self.save_to_memory(self.subtitles)

                    self._play_chunk_wrapper(chunk_data)

                buffer_duration = 0.0
                self.subtitles = ""
                self.current_sentence_keywords = set()

                if event:
                    event.set()
                    with self.lock:
                        self.active_tasks = max(0, self.active_tasks - 1)

                self.audio_queue.task_done()
                continue

            # --- Case 2: Audio Chunk (Correction Logic Here) ---

            # If this is the start of a new sentence, identify the topic
            if incoming_subtitles:
                self.current_sentence_keywords = extract_keywords(incoming_subtitles)

            if self.correction is not None:
                current_keys = self.analyze_ongoing_frame().keys()
                # Normalize config keys to lowercase strings for matching (e.g. "Right Knee" -> "right knee")
                current_keys_lower = {k.lower() for k in current_keys}

                should_abort = False

                # Strategy A: Text mentions specific body parts (e.g. "Fix your knee")
                if self.current_sentence_keywords:
                    is_relevant = False
                    for word in self.current_sentence_keywords:
                        # 1. Direct Partial Match: e.g. "knee" in "left knee"
                        if any(word in key for key in current_keys_lower):
                            is_relevant = True
                            break

                        # 2. Synonym Match: e.g. "back" -> "spine" in "spine alignment"
                        if word in SYNONYMS:
                            synonym = SYNONYMS[word]
                            if any(synonym in key for key in current_keys_lower):
                                is_relevant = True
                                break

                    if not is_relevant:
                        should_abort = True
                        print(
                            f"Aborting: Text topics {self.current_sentence_keywords} not in current errors {list(current_keys)}",
                            file=sys.stderr,
                        )

                # Strategy B: Text is generic (e.g. "Hold it"), check Intersection
                else:
                    # Generic fallback: Play if at least one of the ORIGINAL errors is still present
                    if not (self.correction.keys() & current_keys):
                        should_abort = True
                        print("Aborting: No original errors remain.", file=sys.stderr)

                if should_abort:
                    self._drain_queue_safely()
                    buffer = []
                    buffer_duration = 0.0
                    self.subtitles = ""
                    self.current_sentence_keywords = set()
                    self.audio_queue.task_done()
                    self.move_marty_type_correctiv = None
                    continue

            # --- Buffering & Playback ---
            buffer.append((audio_chunk, incoming_subtitles))
            duration = len(audio_chunk) / SAMPLE_RATE
            buffer_duration += duration

            while buffer_duration >= self.audio_chunk_margin_seconds:
                if self.current_epoch != epoch:
                    buffer = []
                    buffer_duration = 0.0
                    break

                chunk_data, chunk_sub = buffer.pop(0)
                if chunk_sub is not None:
                    self.subtitles = chunk_sub
                    self.save_to_memory(self.subtitles)

                dur = len(chunk_data) / SAMPLE_RATE
                buffer_duration -= dur
                self._play_chunk_wrapper(chunk_data)

            self.audio_queue.task_done()

    def _play_chunk_wrapper(self, chunk):
        if len(chunk) == 0:
            return
        if not self.stream.active:
            self.stream.start()
        while not self.can_i_speak():
            time.sleep(0.1)
        self._play_chunk(self.stream, chunk)

    def _drain_queue_safely(self):
        while not self.audio_queue.empty():
            try:
                item = self.audio_queue.get_nowait()
                if isinstance(item, tuple) and len(item) >= 1:
                    event = item[-1]
                    if event and isinstance(event, threading.Event):
                        event.set()
                        with self.lock:
                            self.active_tasks = max(0, self.active_tasks - 1)
                self.audio_queue.task_done()
            except queue.Empty:
                break

    def _play_chunk(self, stream, chunk):
        duration = len(chunk) / SAMPLE_RATE
        if self.move_marty_enabled:
            if self.move_marty_callback and self.move_marty_type_correctiv is None:
                self.move_marty_callback(duration)
            elif (
                self.move_marty_correctiv and self.move_marty_type_correctiv is not None
            ):
                self.move_marty_correctiv()
                self.move_marty_type_correctiv = None
        stream.write(chunk)

    def _coordinator_worker(self):
        """
        Consumes request_queue, runs Ollama.
        Item format: (epoch, messages, model, completion_event, generate_audio)
        """
        while True:
            request = self.request_queue.get()
            epoch, messages, model, event, generate_audio = request

            if epoch != self.current_epoch:
                if event:
                    event.set()
                self.request_queue.task_done()
                continue

            self._run_ollama_generation(epoch, messages, model, generate_audio)
            self.text_queue.put((epoch, None, event, generate_audio))
            self.request_queue.task_done()

    def _run_ollama_generation(self, epoch, messages, model, generate_audio):
        try:
            stream = ollama.chat(model=model, messages=messages, stream=True)
            buffer = ""
            sentence_endings = re.compile(r"(?<=[.!?])\s+")

            for chunk in stream:
                if epoch != self.current_epoch:
                    break
                content = chunk["message"]["content"]
                buffer += content
                self.generated_text += content
                parts = sentence_endings.split(buffer)

                if len(parts) > 1:
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            self.text_queue.put(
                                (epoch, sentence.strip(), None, generate_audio)
                            )
                    buffer = parts[-1]

            self.generated_text_callback()
            if buffer.strip() and epoch == self.current_epoch:
                self.text_queue.put((epoch, buffer.strip(), None, generate_audio))

        except Exception as e:
            print(f"Ollama Error: {e}", file=sys.stderr)

    def generated_text_callback(self):
        return self.generated_text

    def say(self, messages, model="llama3.1", generate_audio=True):
        """
        Queues a message with the current epoch ID.
        Injects memory of past assistant responses to prevent repetition.
        """
        completion_event = threading.Event()
        with self.lock:
            self.generated_text = ""
            self.active_tasks += 1
            current_epoch = self.current_epoch

            # Create a shallow copy to avoid mutating the original
            final_messages = list(messages)

            # Inject memory before the last message (assumed to be the active User prompt)
            # Structure becomes: [System, ..., Memory (History), ..., User]
            if self.memory and len(final_messages) > 0:
                last_msg = final_messages.pop()
                final_messages.extend(self.memory)
                final_messages.append(last_msg)
            elif self.memory:
                final_messages.extend(self.memory)

        self.request_queue.put(
            (current_epoch, final_messages, model, completion_event, generate_audio)
        )
        return completion_event

    def evaluate_prompt(self, messages, count=20, generate_audio=False):
        print(f"--- Starting Evaluation ({count} runs) ---", file=sys.stderr)
        events = []
        for i in range(count):
            print(f"Queuing Run {i + 1}/{count}...", file=sys.stderr)
            evt = self.say(messages, generate_audio=generate_audio)
            events.append(evt)
        for i, evt in enumerate(events):
            evt.wait()
            print(f"Run {i + 1} completed.", file=sys.stderr)
        print("--- Evaluation Complete ---", file=sys.stderr)

    def intro(self):
        system_instruction = (
            "You are a friendly yoga coach named Marty. Introduce yourself and greet the student warmly. "
            "Keep it to max length of 25 words. No -, use more dots than commas. No questions."
            "Make it sound like a natural conversation."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {
                "role": "user",
                "content": "Greet the student and introduce yourself as their yoga coach.",
            },
        ]
        return self.say(messages, model="llama3.2")

    def start_counter(self):
        completion_event = threading.Event()
        try:
            data, fs = sf.read(COUNTDOWN_FILE_PATH, dtype="float32")
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
                current_epoch = self.current_epoch
            self.audio_queue.put((current_epoch, data, COUNTDOWN_SUBTITLES, None))
            self.audio_queue.put((current_epoch, None, None, completion_event))
        except FileNotFoundError:
            print(f"Error: {COUNTDOWN_FILE_PATH} not found.", file=sys.stderr)
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
            {
                "role": "user",
                "content": f"Pose details: {str(pose['description']['howto'])}",
            },
        ]
        return self.say(messages)

    def load_pose(self, pose):
        system_instruction = (
            "You are a friendly yoga coach. Introduce the pose, briefly describing it while being encouraging. "
            "Keep it to 2 sentences max with max sentence length of 15 words. "
            "At the end say you will demonstrate the pose. "
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Pose details: {str(pose['description'])}"},
        ]
        return self.say(messages)

    def corrective_feedback(self, correction: dict, pose):
        system_instruction = (
            "You are a yoga coach. Receive the corrective feedback. You're inside of a discussion, no mention similar to 'during this pose'. "
            "Keep it to 1 sentences max with max sentence length of 15 words. Be very concise, only useful words. "
            "Don't mention the numbers. No asterisks, No parentheses. "
            "Speak in the present tense and address the student directly without his name. Be creative. "
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "system", "content": f"Pose details: {str(pose['description'])}"},
            {"role": "user", "content": str(correction)},
        ]
        return self.say(messages, model="llama3.2")

    def end_pose_feedback(self, feedbacks):
        self.empty_queues()
        self.correction = None
        self.generated_text = ""
        system_instruction = (
            "You are a friendly yoga coach. Receive the analysis report. "
            "Keep it to 2 sentences max with max sentence length of 20 words. "
            "Highlight a weak points if needed and suggest one improvement tip but be encouraging and positive. "
            "No numbers, no asterisks and no parentheses. "
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
        self.request_queue.join()
        self.text_queue.join()
        self.audio_queue.join()

    def empty_queues(self):
        with self.lock:
            self.current_epoch += 1
            self.active_tasks = 0
            self.current_sentence_keywords = set()

        def drain(q):
            while not q.empty():
                try:
                    item = q.get_nowait()
                    if isinstance(item, tuple) and len(item) >= 1:
                        event = item[-1]
                        if isinstance(event, threading.Event):
                            event.set()
                    q.task_done()
                except queue.Empty:
                    break

        drain(self.request_queue)
        drain(self.text_queue)
        drain(self.audio_queue)
