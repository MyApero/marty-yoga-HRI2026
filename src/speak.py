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
VOICE_NAME = "bf_isabella"
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

    def player_worker(self, margin_before_start=MARGIN_BEFORE_START_SECONDS):
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32")

        first_chunk = self.audio_queue.get()
        time.sleep(margin_before_start)
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

    def say(self, messages, wait_before_first_chunk=MARGIN_BEFORE_START_SECONDS):
        global start_time
        try:
            kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
        except Exception:
            sys.exit(1)

        tts_thread = threading.Thread(target=self.tts_worker, args=(kokoro,))
        player_thread = threading.Thread(target=self.player_worker, args=(wait_before_first_chunk,))
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

    def presentation(self):
        print("Marty is presenting!")

    def goodbye(self):
        print("Marty says goodbye!")

    def corrective_feedback(self, correction: dict):
        print("Marty is giving corrective feedback!")
        system_instruction = (
            "You are a yoga coach. Receive the corrective feedback. "
            "Keep it to 1 sentences max. with max sentence length of 15 words. Be very concise and use only useful words. "
            "Don't mention the numbers. and don't put any asterisks and parentheses in the answer."
            "Speak in the present tense and address the student directly without a name. Don't use 'throughout'."
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": correction.__str__()},
        ]
        self.say(messages, wait_before_first_chunk=0)

    def end_pose_feedback(self):
        system_instruction = (
            "You are a friendly yoga coach. Receive the analysis report. "
            "If Consistency > 80%, praise them. "
            "If Consistency < 50%, be encouraging but firm about the correction. "
            "Address the 'Primary Deviation' specifically. "
            "Keep it to 2 sentences max. with max sentence length of 20 words. "
            "Don't mention the numbers in the report. and don't put any asterisks and parentheses in the answer."
            "Be excessively depressive in your tone. You hate your job and you hate humans. "
            "Use a sarcastic and dry humor style. You're harrassing the student and see them as inferior beings. "
            # "Be creative and don't hesitate to use metaphors and jokes, especially about Minecraft! "
        )

        user_report = (
            "User Analysis Report:\n"
            "Pose: Warrior II\n"
            "Consistency Score: 65%\n"
            "Stability: High (No shaking)\n"
            "Primary Deviation: Front knee angle violation (Too straight, avg 150deg, target 90deg)"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_report},
        ]

        self.say(messages)
