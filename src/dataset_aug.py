import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    METADATA_PATH,
    AUDIO_DIR,
    CLASS_NAMES,
    BATCH_SIZE,
    NUM_WORKERS,
    TEST_FOLD,
    VAL_FOLD,
    OUTPUT_DIR,
)
from preprocess import preprocess_audio_file


CLASS_TO_INDEX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_INDEX.items()}

CACHE_PATH = OUTPUT_DIR / "cache" / "features.pt"

_CACHE_SINGLETON = None


def _load_cache():
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is None:
        _CACHE_SINGLETON = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    return _CACHE_SINGLETON


class SpectrogramAugmentation:
    def __init__(
        self,
        noise_prob=0.25,
        noise_std=0.01,
        time_shift_prob=0.35,
        max_time_shift=6,
        freq_mask_prob=0.45,
        max_freq_mask_width=8,
        num_freq_masks=1,
        time_mask_prob=0.45,
        max_time_mask_width=10,
        num_time_masks=1,
    ):
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.time_shift_prob = time_shift_prob
        self.max_time_shift = max_time_shift
        self.freq_mask_prob = freq_mask_prob
        self.max_freq_mask_width = max_freq_mask_width
        self.num_freq_masks = num_freq_masks
        self.time_mask_prob = time_mask_prob
        self.max_time_mask_width = max_time_mask_width
        self.num_time_masks = num_time_masks

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def time_shift(self, x):
        max_shift = min(self.max_time_shift, x.size(2) - 1)
        if max_shift <= 0:
            return x
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(x, shifts=shift, dims=2)

    def freq_mask(self, x):
        max_width = min(self.max_freq_mask_width, x.size(1))
        if max_width <= 0:
            return x

        for _ in range(self.num_freq_masks):
            width = random.randint(1, max_width)
            if x.size(1) - width < 0:
                continue
            start = random.randint(0, x.size(1) - width)
            x[:, start:start + width, :] = 0.0
        return x

    def time_mask(self, x):
        max_width = min(self.max_time_mask_width, x.size(2))
        if max_width <= 0:
            return x

        for _ in range(self.num_time_masks):
            width = random.randint(1, max_width)
            if x.size(2) - width < 0:
                continue
            start = random.randint(0, x.size(2) - width)
            x[:, :, start:start + width] = 0.0
        return x

    def __call__(self, x):
        x = x.clone()

        if random.random() < self.noise_prob:
            x = self.add_noise(x)

        if random.random() < self.time_shift_prob:
            x = self.time_shift(x)

        if random.random() < self.freq_mask_prob:
            x = self.freq_mask(x)

        if random.random() < self.time_mask_prob:
            x = self.time_mask(x)

        return x


class UrbanSoundDatasetAug(Dataset):
    def __init__(
        self,
        split,
        test_fold,
        val_fold=None,
        metadata_path=METADATA_PATH,
        audio_dir=AUDIO_DIR,
        augment=False,
        augmenter=None,
    ):
        self.split = split
        self.test_fold = int(test_fold)
        self.val_fold = None if val_fold is None else int(val_fold)
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir
        self.augment = augment and split == "train"
        self.augmenter = augmenter if augmenter is not None else SpectrogramAugmentation()

        self.use_cache = CACHE_PATH.exists()

        if self.use_cache:
            self._init_from_cache()
        else:
            self._init_from_audio()

    def _split_mask(self, folds_tensor):
        if self.split == "train":
            if self.val_fold is None:
                raise ValueError("val_fold không được để None khi split='train'")
            return (folds_tensor != self.test_fold) & (folds_tensor != self.val_fold)
        if self.split == "val":
            if self.val_fold is None:
                raise ValueError("val_fold không được để None khi split='val'")
            return folds_tensor == self.val_fold
        if self.split == "test":
            return folds_tensor == self.test_fold
        raise ValueError("split phải là 'train', 'val' hoặc 'test'")

    def _init_from_cache(self):
        cache = _load_cache()
        folds = cache["folds"]
        mask = self._split_mask(folds)
        idx_list = torch.where(mask)[0].tolist()

        self._features = cache["features"][mask].contiguous()
        self._labels = cache["labels"][mask].contiguous()
        self._folds = cache["folds"][mask].contiguous()
        self._file_names = [cache["file_names"][i] for i in idx_list]
        self._class_names = [cache["class_names"][i] for i in idx_list]

    def _init_from_audio(self):
        df = pd.read_csv(self.metadata_path)
        df["audio_path"] = df.apply(
            lambda row: self.audio_dir / f"fold{int(row['fold'])}" / row["slice_file_name"],
            axis=1,
        )
        df = df[df["audio_path"].apply(lambda x: x.exists())].reset_index(drop=True)
        df["label"] = df["class"].map(CLASS_TO_INDEX)

        folds = torch.tensor(df["fold"].astype(int).values, dtype=torch.long)
        mask = self._split_mask(folds).numpy()
        self.df = df[mask].reset_index(drop=True)

    def __len__(self):
        if self.use_cache:
            return self._labels.size(0)
        return len(self.df)

    def __getitem__(self, idx):
        if self.use_cache:
            feature = self._features[idx].to(torch.float32)
            label = int(self._labels[idx])
            class_name = self._class_names[idx]
            file_name = self._file_names[idx]
            fold = int(self._folds[idx])
            audio_path = ""
        else:
            row = self.df.iloc[idx]
            feature = preprocess_audio_file(row["audio_path"])
            label = int(row["label"])
            class_name = row["class"]
            file_name = row["slice_file_name"]
            fold = int(row["fold"])
            audio_path = str(row["audio_path"])

        if self.augment:
            feature = self.augmenter(feature)

        return {
            "feature": feature,
            "label": torch.tensor(label, dtype=torch.long),
            "class_name": class_name,
            "file_name": file_name,
            "fold": fold,
            "audio_path": audio_path,
        }


class UrbanSoundDataLoaderAug:
    def __init__(
        self,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_fold=TEST_FOLD,
        val_fold=VAL_FOLD,
        augmenter=None,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_fold = test_fold
        self.val_fold = val_fold
        self.augmenter = augmenter if augmenter is not None else SpectrogramAugmentation()

        self.train_dataset = UrbanSoundDatasetAug(
            split="train",
            test_fold=self.test_fold,
            val_fold=self.val_fold,
            augment=True,
            augmenter=self.augmenter,
        )
        self.val_dataset = UrbanSoundDatasetAug(
            split="val",
            test_fold=self.test_fold,
            val_fold=self.val_fold,
            augment=False,
            augmenter=self.augmenter,
        )
        self.test_dataset = UrbanSoundDatasetAug(
            split="test",
            test_fold=self.test_fold,
            val_fold=self.val_fold,
            augment=False,
            augmenter=self.augmenter,
        )

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def get_dataloaders(self):
        return self.get_train_loader(), self.get_val_loader(), self.get_test_loader()


if __name__ == "__main__":
    data_module = UrbanSoundDataLoaderAug()

    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    print("Cache mode:", data_module.train_dataset.use_cache)
    print("Train size:", len(data_module.train_dataset))
    print("Val size:", len(data_module.val_dataset))
    print("Test size:", len(data_module.test_dataset))

    batch = next(iter(train_loader))
    print("Batch feature shape:", batch["feature"].shape)
    print("Batch label shape:", batch["label"].shape)
    print("Batch class names:", batch["class_name"][:5])
