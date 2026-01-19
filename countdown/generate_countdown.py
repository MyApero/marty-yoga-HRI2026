from kokoro import KPipeline
import soundfile as sf
import torch
pipeline = KPipeline(lang_code='b')
text = '''3, 2, 1'''
generator = pipeline(text, voice='am_michael')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    sf.write('countdown.wav', audio, 24000)
