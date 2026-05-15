from pathlib import Path
import argparse
import time

import pandas as pd
import torch
from tqdm import tqdm

from config import (
    METADATA_PATH,
    AUDIO_DIR,
    CLASS_NAMES,
    OUTPUT_DIR,
)
from preprocess import preprocess_audio_file


CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_PATH = CACHE_DIR / "features.pt"

CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def build_cache(cache_path=CACHE_PATH, force=False, dtype=torch.float16):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force:
        print(f"Cache đã tồn tại: {cache_path}")
        print("Dùng --force để build lại.")
        return cache_path

    df = pd.read_csv(METADATA_PATH)
    df["audio_path"] = df.apply(
        lambda row: AUDIO_DIR / f"fold{int(row['fold'])}" / row["slice_file_name"],
        axis=1,
    )
    df = df[df["audio_path"].apply(lambda p: p.exists())].reset_index(drop=True)

    print(f"Sẽ cache {len(df)} samples vào {cache_path}")

    features_list = []
    labels = []
    folds = []
    file_names = []
    class_names = []

    start = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching log-Mel"):
        feature = preprocess_audio_file(row["audio_path"])
        features_list.append(feature.to(dtype))
        labels.append(CLASS_TO_INDEX[row["class"]])
        folds.append(int(row["fold"]))
        file_names.append(row["slice_file_name"])
        class_names.append(row["class"])

    features = torch.stack(features_list)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    folds_tensor = torch.tensor(folds, dtype=torch.long)

    payload = {
        "features": features,
        "labels": labels_tensor,
        "folds": folds_tensor,
        "file_names": file_names,
        "class_names": class_names,
    }

    torch.save(payload, cache_path)

    elapsed = time.time() - start
    size_mb = cache_path.stat().st_size / (1024 ** 2)
    print()
    print(f"Done in {elapsed:.1f}s")
    print(f"Saved to {cache_path} ({size_mb:.1f} MB)")
    print(f"Feature tensor: shape={tuple(features.shape)}, dtype={features.dtype}")

    return cache_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Build lại dù cache đã tồn tại")
    args = parser.parse_args()

    build_cache(force=args.force)


if __name__ == "__main__":
    main()
