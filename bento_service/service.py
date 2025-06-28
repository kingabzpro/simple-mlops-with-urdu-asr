import bentoml, torch, transformers
from bentoml.io import File, JSON

MODEL_TAG = "whisper_urdu_service:latest"
model_ref = bentoml.models.get(MODEL_TAG)
processor = transformers.WhisperProcessor.from_pretrained(model_ref.path)
model = transformers.WhisperForConditionalGeneration.from_pretrained(model_ref.path).to(
    "cpu"
)

svc = bentoml.Service("urdu_asr", runners=[])


@svc.api(input=File(), output=JSON())
def transcribe(audio_file):
    import soundfile as sf
    import io

    data, sr = sf.read(io.BytesIO(audio_file.read()), dtype="float32")
    input_f = processor.feature_extractor(data, sampling_rate=sr, return_tensors="pt")
    pred_ids = model.generate(input_f.input_features)
    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return {"transcript": text}
