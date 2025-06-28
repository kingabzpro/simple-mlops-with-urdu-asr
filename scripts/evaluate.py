import datasets, jiwer, torch, transformers
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pathlib import Path
import json, argparse

MODEL_DIR = Path("checkpoints/best")
processor = WhisperProcessor.from_pretrained(MODEL_DIR)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def infer(batch):
    inputs = processor.feature_extractor(
        batch["audio"], sampling_rate=16000, return_tensors="pt"
    )
    with torch.no_grad():
        pred_ids = model.generate(inputs.input_features.to(model.device))
    batch["prediction"] = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return batch


def main(split="test", save_to="metrics.json"):
    ds = datasets.load_from_disk(f"data/processed/{split}").map(infer)
    metric = jiwer.wer(ds["sentence"], ds["prediction"])
    Path(save_to).write_text(json.dumps({"wer": metric}))
    print(f"WER: {metric:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--save_to", default="metrics.json")
    main(**vars(parser.parse_args()))
