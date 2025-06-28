"""End‑to‑end DAG using Prefect 3 (no Prefect Cloud)."""

from prefect import flow, task
import subprocess, os


@task(retries=2, retry_delay_seconds=60)
def download():
    subprocess.check_call(["python", "scripts/download_dataset.py", "--split", "train"])
    subprocess.check_call(
        ["python", "scripts/download_dataset.py", "--split", "validation"]
    )


@task(retries=2)
def preprocess():
    subprocess.check_call(
        [
            "python",
            "scripts/preprocess_dataset.py",
            "--in_path",
            "data/raw/train.parquet",
            "--out_path",
            "data/processed/train",
        ]
    )
    subprocess.check_call(
        [
            "python",
            "scripts/preprocess_dataset.py",
            "--in_path",
            "data/raw/validation.parquet",
            "--out_path",
            "data/processed/validation",
        ]
    )


@task
def train():
    subprocess.check_call(["python", "scripts/train.py"])


@task
def evaluate():
    subprocess.check_call(["python", "scripts/evaluate.py"])


@task
def build_bento():
    subprocess.check_call(["bentoml", "build", "."])
    subprocess.check_call(
        ["bentoml", "push", "urdu-asr:latest"]
    )  # requires env vars


@flow
def asr_pipeline():
    download()
    preprocess()
    train()
    evaluate()
    build_bento()


if __name__ == "__main__":
    asr_pipeline()
