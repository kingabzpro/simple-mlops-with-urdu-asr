"""Downloads the Common Voice v17 Urdu split with the Hugging Face `datasets` API."""

import os
from datasets import load_dataset
from pathlib import Path


def main(split: str = "train", save_dir: str = "data/raw"):
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "ur",
        split=split,
        token=os.getenv("HF_TOKEN"),
    )
    out = Path(save_dir) / f"{split}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out)
    print(f"Saved {len(ds):,} samples → {out}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--save_dir", default="data/raw")
    main(**vars(p.parse_args()))
