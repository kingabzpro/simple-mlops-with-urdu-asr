"""Cleans transcripts, resamples audio to 16 kHz mono FLAC, and keeps only Urdu text."""

import datasets, torchaudio, re, io, soundfile as sf
from pathlib import Path

URDU_CHARS = r"[\u0600-\u06FF\s]"
RE_URDU = re.compile(f"^{URDU_CHARS}+$")


def _valid(example):
    return bool(RE_URDU.match(example["sentence"]))


def _process(example):
    speech_array, sr = torchaudio.load(io.BytesIO(example["audio"]["array"]))
    speech_array = torchaudio.functional.resample(speech_array, sr, 16000)
    flac_bytes = io.BytesIO()
    sf.write(flac_bytes, speech_array.squeeze().numpy(), 16000, format="FLAC")
    example.update(audio=flac_bytes.getvalue())
    return example


def main(in_path="data/raw/train.parquet", out_path="data/processed/train"):
    ds = datasets.Dataset.from_parquet(in_path).filter(_valid)
    ds = ds.map(_process, num_proc=8)
    ds.save_to_disk(out_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--in_path", default="data/raw/train.parquet")
    p.add_argument("--out_path", default="data/processed/train")
    main(**vars(p.parse_args()))
