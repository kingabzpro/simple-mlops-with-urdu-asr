"""Fine‑tunes Whisper‑large‑v3 on processed Urdu and logs everything to MLflow."""

import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import torch
import transformers
import datasets
import numpy as np
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "model_id": os.getenv("MODEL_ID", "openai/whisper-large-v3"),
    "language": "urdu",
    "task": "transcribe",
    "data_dir": Path("data/processed"),
    "output_dir": Path("checkpoints"),
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "mlruns"),
    "experiment_name": "whisper-urdu-asr",
    "run_name": f"whisper-urdu-{int(time.time())}",
    "seed": 42,
    "num_proc": int(os.getenv("NUM_PROC", "4")),
}

# Set up MLflow
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
mlflow.set_experiment(CONFIG["experiment_name"])


class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text tasks."""
    
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Split inputs and labels since they have different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS token is appended in previous tokenization step,
        # cut BOS token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def setup_model_and_processor() -> tuple:
    """Load and setup model and processor."""
    logger.info(f"Loading model: {CONFIG['model_id']}")
    
    processor = WhisperProcessor.from_pretrained(
        CONFIG['model_id'], 
        language=CONFIG['language'], 
        task=CONFIG['task']
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(CONFIG['model_id'])
    
    # Set special tokens
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    
    return model, processor

def load_and_prepare_data(processor) -> tuple:
    """Load and preprocess datasets."""
    logger.info("Loading and preprocessing datasets")
    
    def preprocess_function(examples):
        # Extract audio arrays
        audios = [audio for audio in examples["audio"]]
        
        # Compute input features
        inputs = processor.feature_extractor(
            audios, 
            sampling_rate=16000, 
            return_tensors="np"
        )
        
        # Encode target text to label ids
        labels = processor.tokenizer(
            examples["sentence"],
            truncation=True,
            padding=False,
            return_tensors="np"
        )
        
        examples["input_features"] = inputs.input_features.tolist()
        examples["labels"] = labels.input_ids.tolist()
        
        return examples
    
    try:
        train_ds = datasets.load_from_disk(CONFIG["data_dir"] / "train")
        eval_ds = datasets.load_from_disk(CONFIG["data_dir"] / "validation")
        
        logger.info(f"Train dataset size: {len(train_ds)}")
        logger.info(f"Validation dataset size: {len(eval_ds)}")
        
        # Preprocess datasets
        train_ds = train_ds.map(
            preprocess_function,
            batched=True,
            num_proc=CONFIG["num_proc"],
            remove_columns=["audio", "sentence"]
        )
        
        eval_ds = eval_ds.map(
            preprocess_function,
            batched=True,
            num_proc=CONFIG["num_proc"],
            remove_columns=["audio", "sentence"]
        )
        
        return train_ds, eval_ds
        
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        raise

def get_training_arguments() -> TrainingArguments:
    """Create training arguments with optimized settings."""
    
    # Create output directory
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    
    return TrainingArguments(
        output_dir=str(CONFIG["output_dir"]),
        per_device_train_batch_size=int(os.getenv("TRAIN_BATCH_SIZE", "16")),
        per_device_eval_batch_size=int(os.getenv("EVAL_BATCH_SIZE", "8")),
        gradient_accumulation_steps=int(os.getenv("GRAD_ACCUM_STEPS", "2")),
        learning_rate=float(os.getenv("LEARNING_RATE", "1e-5")),
        warmup_steps=int(os.getenv("WARMUP_STEPS", "500")),
        num_train_epochs=int(os.getenv("NUM_EPOCHS", "3")),
        evaluation_strategy="steps",
        eval_steps=int(os.getenv("EVAL_STEPS", "500")),
        save_steps=int(os.getenv("SAVE_STEPS", "500")),
        logging_steps=int(os.getenv("LOGGING_STEPS", "50")),
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["mlflow"] if os.getenv("USE_MLFLOW", "true").lower() == "true" else [],
        run_name=CONFIG["run_name"],
        seed=CONFIG["seed"],
        data_seed=CONFIG["seed"],
        logging_dir=str(CONFIG["output_dir"] / "logs"),
    )

def main():
    """Main training function."""
    logger.info("Starting Whisper fine-tuning for Urdu ASR")
    
    # Set seeds for reproducibility
    transformers.set_seed(CONFIG["seed"])
    
    try:
        with mlflow.start_run(run_name=CONFIG["run_name"]) as run:
            logger.info(f"MLflow run ID: {run.info.run_id}")
            
            # Log configuration
            mlflow.log_params(CONFIG)
            
            # Setup model and processor
            model, processor = setup_model_and_processor()
            
            # Load and prepare data
            train_ds, eval_ds = load_and_prepare_data(processor)
            
            # Create data collator
            data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)
            
            # Get training arguments
            training_args = get_training_arguments()
            
            # Log training arguments
            mlflow.log_params(training_args.to_dict())
            
            # Check for previous checkpoints
            last_checkpoint = None
            if CONFIG["output_dir"].exists():
                last_checkpoint = get_last_checkpoint(str(CONFIG["output_dir"]))
                if last_checkpoint:
                    logger.info(f"Found checkpoint: {last_checkpoint}")
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=data_collator,
                tokenizer=processor.feature_extractor,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Start training
            logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            
            # Log training metrics
            mlflow.log_metrics(train_result.metrics)
            
            # Save best model
            best_model_dir = CONFIG["output_dir"] / "best"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            
            trainer.save_model(str(best_model_dir))
            processor.save_pretrained(str(best_model_dir))
            
            logger.info(f"Best model saved to: {best_model_dir}")
            
            # Log model to MLflow
            try:
                mlflow.transformers.log_model(
                    transformers_model={
                        "model": trainer.model,
                        "tokenizer": processor
                    },
                    artifact_path="model",
                    task="automatic-speech-recognition",
                    registered_model_name="whisper-urdu-asr"
                )
                logger.info("Model logged to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {str(e)}")
            
            # Log final metrics and tags
            mlflow.set_tags({
                "model_name": "whisper-urdu-v3-finetuned",
                "language": CONFIG["language"],
                "task": CONFIG["task"],
                "base_model": CONFIG["model_id"],
                "status": "completed"
            })
            
            # Save training summary
            summary = {
                "model_id": CONFIG["model_id"],
                "train_samples": len(train_ds),
                "eval_samples": len(eval_ds),
                "final_train_loss": train_result.training_loss,
                "best_model_path": str(best_model_dir),
                "mlflow_run_id": run.info.run_id
            }
            
            summary_file = CONFIG["output_dir"] / "training_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("Training completed successfully!")
            logger.info(f"Training summary saved to: {summary_file}")
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        # Log failure to MLflow
        if mlflow.active_run():
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error", str(e))
        raise


if __name__ == "__main__":
    main()
