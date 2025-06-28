"""Gradio interface for Urdu ASR service."""

import os
import time
import logging
from typing import Optional, Tuple
import io

import gradio as gr
import requests
import soundfile as sf
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "endpoint": os.getenv(
        "BENTO_ENDPOINT", 
        "https://<YOUR_BENTOCLOUD_NAMESPACE>.bentoml.app/transcribe"
    ),
    "health_endpoint": os.getenv(
        "BENTO_HEALTH_ENDPOINT",
        "https://<YOUR_BENTOCLOUD_NAMESPACE>.bentoml.app/health"
    ),
    "timeout": int(os.getenv("REQUEST_TIMEOUT", "30")),
    "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "25")),
}

def check_service_health() -> Tuple[bool, str]:
    """Check if the BentoML service is healthy."""
    try:
        response = requests.get(CONFIG["health_endpoint"], timeout=5)
        if response.status_code == 200:
            return True, "Service is healthy"
        else:
            return False, f"Service returned status code: {response.status_code}"
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

def transcribe_audio(audio: Optional[Tuple[int, np.ndarray]]) -> str:
    """Transcribe audio using the BentoML service.
    
    Args:
        audio: Tuple of (sample_rate, audio_data) from Gradio
        
    Returns:
        Transcription text or error message
    """
    if audio is None:
        return "❌ No audio provided. Please record or upload an audio file."
    
    try:
        sample_rate, audio_data = audio
        
        # Convert to float32 and normalize
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0
        
        # Ensure mono audio
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Create WAV bytes
        wav_bytes = io.BytesIO()
        sf.write(wav_bytes, audio_data, sample_rate, format="WAV")
        wav_bytes.seek(0)
        
        # Check file size
        file_size_mb = len(wav_bytes.getvalue()) / (1024 * 1024)
        if file_size_mb > CONFIG["max_file_size_mb"]:
            return f"❌ File too large ({file_size_mb:.1f}MB). Maximum size is {CONFIG['max_file_size_mb']}MB."
        
        logger.info(f"Sending audio file ({file_size_mb:.2f}MB) to transcription service")
        
        # Make request to BentoML service
        start_time = time.time()
        response = requests.post(
            CONFIG["endpoint"],
            files={"audio_file": ("audio.wav", wav_bytes, "audio/wav")},
            timeout=CONFIG["timeout"]
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            if "error" in result:
                error_msg = result.get("error", "Unknown error")
                error_type = result.get("error_type", "unknown")
                return f"❌ Error ({error_type}): {error_msg}"
            
            transcript = result.get("transcript", "")
            server_time = result.get("processing_time", 0)
            
            if not transcript.strip():
                return "⚠️ No speech detected in the audio. Please try with clearer audio."
            
            # Add processing time info
            time_info = f"\n\n⏱️ Processing time: {processing_time:.2f}s (client) + {server_time:.2f}s (server)"
            return f"✅ **Transcription:**\n\n{transcript}{time_info}"
        
        else:
            return f"❌ Service error: HTTP {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return f"❌ Request timeout ({CONFIG['timeout']}s). The audio file might be too long or the service is busy."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to transcription service. Please check if the service is running."
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return f"❌ Unexpected error: {str(e)}"

def create_interface():
    """Create and configure the Gradio interface."""
    
    # Check service health on startup
    is_healthy, health_msg = check_service_health()
    health_status = f"🟢 {health_msg}" if is_healthy else f"🔴 {health_msg}"
    
    with gr.Blocks(
        title="Urdu ASR - Whisper v3",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 800px !important;
            margin: auto !important;
        }
        .transcription-output {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # 🎤 Urdu Automatic Speech Recognition
            
            Upload an audio file or record your voice to get Urdu transcription using fine-tuned Whisper-v3.
            
            **Supported formats:** WAV, MP3, FLAC, M4A, OGG  
            **Maximum file size:** 25MB  
            **Language:** Urdu
            """
        )
        
        with gr.Row():
            gr.Markdown(f"**Service Status:** {health_status}")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="Audio Input",
                    format="wav"
                )
                
                transcribe_btn = gr.Button(
                    "🎯 Transcribe Audio",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column():
                output_text = gr.Textbox(
                    label="Transcription Result",
                    placeholder="Your transcription will appear here...",
                    lines=10,
                    max_lines=20,
                    elem_classes=["transcription-output"]
                )
        
        with gr.Row():
            gr.Examples(
                examples=[
                    ["example_audio.wav"] if Path("example_audio.wav").exists() else [],
                ],
                inputs=audio_input,
                label="Example Audio Files"
            )
        
        with gr.Accordion("ℹ️ Usage Tips", open=False):
            gr.Markdown(
                """
                ### Tips for better transcription:
                
                - **Clear audio:** Use good quality recordings with minimal background noise
                - **Proper volume:** Ensure the audio is not too quiet or too loud
                - **Language:** Speak clearly in Urdu for best results
                - **Length:** Shorter clips (under 30 seconds) typically work better
                - **Format:** WAV files usually provide the best quality
                
                ### Troubleshooting:
                
                - If you get connection errors, check that the BentoML service is running
                - For large files, try splitting them into smaller segments
                - If transcription is empty, the audio might be too quiet or unclear
                """
            )
        
        # Event handlers
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[output_text],
            api_name="transcribe"
        )
        
        audio_input.change(
            fn=lambda: "🎵 Audio uploaded. Click 'Transcribe Audio' to process.",
            outputs=[output_text]
        )
    
    return demo

def main():
    """Main function to launch the Gradio app."""
    logger.info("Starting Gradio interface for Urdu ASR")
    logger.info(f"Using endpoint: {CONFIG['endpoint']}")
    
    demo = create_interface()
    
    # Launch configuration
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.getenv("PORT", "7860")),
        "share": os.getenv("GRADIO_SHARE", "false").lower() == "true",
        "debug": os.getenv("DEBUG", "false").lower() == "true",
    }
    
    demo.launch(**launch_kwargs)

if __name__ == "__main__":
    main()
