from kokoro import KPipeline
import soundfile as sf
pipeline = KPipeline(lang_code='b', repo_id='hexgrad/Kokoro-82M')
text = '''3, 2, 1'''
generator = pipeline(text, voice='am_michael')
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    sf.write('countdown.wav', audio, 24000)
