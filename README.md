# Urdu ASR – End‑to‑End MLOps

This repo fine‑tunes Whisper‑large‑v3 on Common Voice Urdu, governs the pipeline with Prefect, tracks metrics in MLflow, and serves the model via BentoCloud.

```sh
# Run everything locally
prefect run --load asr_pipeline
# Debug Gradio locally (requires port‑forwarding to Bento Cloud)
python gradio_app/app.py
```
