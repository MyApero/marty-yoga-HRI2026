#!/bin/bash
# Quick Test - Verify your installation works

echo "🧪 Testing Meditation Audio Generator..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Create test segments
echo "Creating test audio..."
echo "3..." > test_seg1.txt
echo "2..." > test_seg2.txt
echo "1..." > test_seg3.txt
echo "Hold!!!" > test_seg4.txt
# echo $(ollama run llama3.1 "Present Yourself") > test_seg2.txt
# echo "We're gonna start by ." > test_seg2.txt

# Generate audio
kokoro-tts test_seg1.txt test_seg1.wav --voice am_michael --speed 1 --lang en-gb 2>&1 | grep -v "Processing"
kokoro-tts test_seg2.txt test_seg2.wav --voice am_michael --speed 1 --lang en-gb 2>&1 | grep -v "Processing"
kokoro-tts test_seg3.txt test_seg3.wav --voice am_michael --speed 1 --lang en-gb 2>&1 | grep -v "Processing"
kokoro-tts test_seg4.txt test_seg4.wav --voice am_michael --speed 1 --lang en-gb 2>&1 | grep -v "Processing"

# Merge with pause
python utils/add_silence_between.py test_seg1.wav test_seg2.wav temp1.wav --pause 0.53
python utils/add_silence_between.py temp1.wav test_seg3.wav temp2.wav --pause 0.53
python utils/add_silence_between.py temp2.wav test_seg4.wav test_output.wav --pause 0.53

# Cleanup
# rm test_seg*.txt test_seg*.wav

echo ""
echo "✅ Test successful! Audio file created: test_output.wav"
echo "🎧 Play it to verify everything works"
echo ""
echo "Next steps:"
echo "  1. Visit https://chatgpt.com/g/g-68ef01a860888191a359f20813bc89e6-meditation-scripts"
echo "  2. Generate your first meditation script"
echo "  3. Run it and enjoy!"

