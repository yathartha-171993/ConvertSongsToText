import streamlit as st
from transformers import pipeline
import tempfile
import json

# ✅ Load the Whisper pipeline
@st.cache_resource
def load_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        chunk_length_s=30,
        return_timestamps=True
    )

asr = load_model()

# ✅ Language name mapping
LANG_NAME = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    # add more if needed
}

# ✅ Streamlit UI
st.title("🎵 Convert Songs to Text")
st.markdown("Upload an audio file to transcribe lyrics using OpenAI Whisper")

# 📁 File uploader
uploaded_file = st.file_uploader("Upload your audio file (.wav, .mp3)", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    with st.spinner("Transcribing... this may take a moment ⏳"):
        result = asr(temp_path)

    # 🗣️ Language info
    language = result.get("language", "unknown")
    language_name = LANG_NAME.get(language, language)
    st.success(f"Detected Language: **{language_name}**")

    # 📝 Show transcript with timestamps
    st.markdown("### Transcription with Timestamps")
    for chunk in result.get("chunks", []):
        time = round(chunk['timestamp'][0], 2)
        text = chunk['text'].strip()
        st.markdown(f"**[{time} sec]**: {text}")

    # 📤 Download JSON
    transcript_json = json.dumps(result, indent=2)
    st.download_button(
        label="Download Full Transcript (JSON)",
        data=transcript_json,
        file_name="transcript.json",
        mime="application/json"
    )
