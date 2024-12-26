"""
Audio Transcription Application using Whisper
Supports both microphone recording and file uploads.
"""

# Standard library imports
import os
import tempfile
from typing import Optional, Tuple, Union

# Third-party imports
import numpy as np
import streamlit as st
import whisper
from scipy.io.wavfile import write

# Constants
SAMPLE_RATE = 16000
TEMP_AUDIO_FILENAME = "temp_audio.wav"
SUPPORTED_AUDIO_TYPES = ['wav', 'mp3', 'ogg','opus']

def save_audio(audio_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """
    Save uploaded audio file to a temporary WAV file.
    
    Args:
        audio_file: Uploaded audio file
    
    Returns:
        str: Path to the saved temporary file
    """
    temp_dir = tempfile.gettempdir()  # Use system temp directory
    temp_path = os.path.join(temp_dir, TEMP_AUDIO_FILENAME)
    
    with open(temp_path, 'wb') as f:
        f.write(audio_file.getvalue())
    return temp_path

def create_ui_columns() -> Tuple[st.delta_generator.DeltaGenerator, 
                                st.delta_generator.DeltaGenerator]:
    """Create and return the main UI columns."""
    return st.columns([2, 1])

def display_instructions() -> None:
    """Display application instructions."""
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload an audio file (.wav, .mp3, or .ogg format)
    2. Wait for the transcription to complete
    3. Your transcription will appear below
    """)

def handle_file_upload(model: whisper.Whisper) -> None:
    """Handle audio file upload and transcription."""
    st.subheader("📁 Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=SUPPORTED_AUDIO_TYPES)
    
    if uploaded_file:
        with st.spinner("Transcribing..."):
            try:
                temp_path = save_audio(uploaded_file)
                result = model.transcribe(temp_path)
                st.session_state.transcription = result["text"]
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

def display_output() -> None:
    """Display transcription output and download options."""
    if st.session_state.transcription:
        st.markdown("### Transcription")
        st.write(st.session_state.transcription)
        st.code(st.session_state.transcription, language=None)
        
        st.download_button(
            "Save as TXT",
            st.session_state.transcription,
            file_name="transcription.txt",
            mime="text/plain",
            use_container_width=False
        )

def initialize_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if 'transcription' not in st.session_state:
        st.session_state.transcription = ""
    if 'model' not in st.session_state:
        st.session_state.model = whisper.load_model("base")

def load_whisper_model(model_name: str) -> whisper.Whisper:
    """
    Load a Whisper model if it's different from the currently loaded model.
    
    Args:
        model_name: Name of the Whisper model to load
        
    Returns:
        whisper.Whisper: Loaded model
    """
    if not hasattr(st.session_state, 'current_model_name') or \
       st.session_state.current_model_name != model_name:
        with st.spinner(f"Loading {model_name} model..."):
            st.session_state.model = whisper.load_model(model_name)
            st.session_state.current_model_name = model_name
    return st.session_state.model

def main() -> None:
    """Main application function."""
    # Set page configuration
    st.set_page_config(
        page_title="Audio Transcription App",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("VAM Audio Transcription")
    st.write("Convert your speech to text using OpenAI's Whisper model")
    
    initialize_session_state()
    col1, col2 = create_ui_columns()
    
    with col1:
        display_instructions()
        
        # Model selection
        model_name = st.selectbox(
            "Select Model",
            ["tiny", "base"],
            help="Larger models are more accurate but slower"
        )
        st.session_state.model = load_whisper_model(model_name)
        
        handle_file_upload(st.session_state.model)
        
        if st.session_state.transcription:
            st.markdown("### Transcription")
            st.write(st.session_state.transcription)
    
    with col2:
        st.markdown("### Output")
        display_output()

if __name__ == "__main__":
    main()