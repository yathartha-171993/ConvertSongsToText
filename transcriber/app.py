from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import tempfile

app = Flask(__name__)

# ✅ Load Whisper pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,           # Enables chunking for long audio
    return_timestamps=True
)

# ✅ Language name mapping
LANG_NAME = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    # add more as needed
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    file = request.files['audio']
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        file.save(temp.name)

        # 🎙️ Transcribe with timestamps
        result = asr(temp.name)

        language = result.get('language', 'unknown')
        language_name = LANG_NAME.get(language, language)

        segments = []
        for chunk in result['chunks']:
            timestamp = chunk['timestamp'][0]  # start time
            text = chunk['text'].strip()
            segments.append({"time": timestamp, "text": text})

    return jsonify({"language": language_name, "segments": segments})

if __name__ == "__main__":
    app.run(debug=True)
