"""Fine‑tunes Whisper‑large‑v3 on processed Urdu and logs everything to MLflow."""

import os, mlflow, torch, transformers, datasets
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
)
from pathlib import Path

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment("whisper-urdu-asr")

MODEL_ID = "openai/whisper-large-v3"


def load_data(split="train"):
    ds = datasets.load_from_disk(f"data/processed/{split}")
    return ds


def main():
    with mlflow.start_run() as run:
        processor = WhisperProcessor.from_pretrained(
            MODEL_ID, language="Urdu", task="transcribe"
        )
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

        def preprocess(ex):
            ex["input_features"] = processor.feature_extractor(
                ex["audio"], sampling_rate=16000
            ).input_features[0]
            ex["labels"] = processor.tokenizer(ex["sentence"]).input_ids
            return ex

        train_ds = load_data("train").map(
            preprocess, remove_columns=["sentence", "audio"], num_proc=4
        )
        eval_ds = load_data("validation").map(
            preprocess, remove_columns=["sentence", "audio"], num_proc=4
        )

        args = TrainingArguments(
            output_dir="checkpoints",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            fp16=True,
            learning_rate=1e-5,
            warmup_steps=500,
            logging_steps=50,
            save_total_limit=2,
            num_train_epochs=3,
            report_to=["mlflow"],
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds
        )
        trainer.train()
        mlflow.pyfunc.log_model(
            "model", python_model=None, artifacts={"model_dir": "checkpoints"}
        )
        mlflow.log_params(args.to_dict())
        mlflow.set_tag("model_name", "whisper-urdu-v3-finetuned")


if __name__ == "__main__":
    main()
