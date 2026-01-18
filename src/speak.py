import ollama
import sounddevice as sd
from kokoro_onnx import Kokoro
import threading
import queue
import re
import time
import sys

# --- Configuration ---
MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"
VOICE_NAME = "am_echo"
VOICE_LANG = "en-gb"
# bf_alice, bf_emma, bf_isabella, bf_lily, bm_daniel, bm_fable, bm_george, bm_lewis
VOICE_SPEED = 0.9
SAMPLE_RATE = 24000
MARGIN_BEFORE_START_SECONDS = 2.0


class Speak:
    def __init__(self, move_marty_callback):
        # --- Global State ---
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.start_time = 0
        self.move_marty_callback = move_marty_callback

    def tts_worker(self, kokoro):
        while True:
            text = self.text_queue.get()
            if text is None:
                self.audio_queue.put(None)
                break
            try:
                audio, _ = kokoro.create(
                    text, voice=VOICE_NAME, speed=VOICE_SPEED, lang=VOICE_LANG
                )
                if len(audio) > 0:
                    self.audio_queue.put(audio)
            except Exception as e:
                print(f"TTS Error: {e}", file=sys.stderr)
            self.text_queue.task_done()

    def player_worker(self):
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")

        first_chunk = self.audio_queue.get()
        time.sleep(MARGIN_BEFORE_START_SECONDS)
        stream.start()

        chunk_index = 0

        while True:
            # get() blocks until data is ready.
            # 'now' is effectively the "Ready Time" of the current chunk
            if chunk_index == 0:
                audio_chunk = first_chunk
            else:
                audio_chunk = self.audio_queue.get()

            if audio_chunk is None:
                break

            # Duration of the current audio chunk in seconds
            chunk_duration = len(audio_chunk) / SAMPLE_RATE
            self.move_marty_callback(chunk_duration)

            stream.write(audio_chunk)
            self.audio_queue.task_done()
            chunk_index += 1

        stream.stop()
        stream.close()

    def say(self, messages):
        global start_time
        try:
            kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
        except Exception:
            sys.exit(1)

        tts_thread = threading.Thread(target=self.tts_worker, args=(kokoro,))
        player_thread = threading.Thread(target=self.player_worker)
        tts_thread.start()
        player_thread.start()

        # --- START TIMER ---
        start_time = time.perf_counter()

        stream = ollama.chat(
            model="llama3.1",
            messages=messages,
            stream=True,
        )

        buffer = ""
        sentence_endings = re.compile(r"(?<=[.!?])\s+")

        for chunk in stream:
            content = chunk["message"]["content"]
            # Print to stderr to keep stdout clean for data
            print(content, end='', file=sys.stderr, flush=True)

            buffer += content
            parts = sentence_endings.split(buffer)

            if len(parts) > 1:
                for sentence in parts[:-1]:
                    if sentence.strip():
                        self.text_queue.put(sentence.strip())
                buffer = parts[-1]

        if buffer.strip():
            self.text_queue.put(buffer.strip())

        self.text_queue.put(None)
        tts_thread.join()
        player_thread.join()
