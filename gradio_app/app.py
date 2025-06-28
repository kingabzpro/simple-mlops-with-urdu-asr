import gradio as gr, requests, io, soundfile as sf

ENDPOINT = "https://<YOUR_BENTOCLOUD_NAMESPACE>.bentoml.app/urdu_asr/transcribe"


def transcribe(audio):
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, audio[1], 16000, format="WAV")
    wav_bytes.seek(0)
    resp = requests.post(ENDPOINT, files={"audio_file": wav_bytes})
    return resp.json().get("transcript", "error")


ui = gr.Interface(
    transcribe,
    gr.Audio(sources=["upload", "microphone"]),
    "text",
    title="Urdu ASR ‑ Whisper v3",
)
ui.launch()
