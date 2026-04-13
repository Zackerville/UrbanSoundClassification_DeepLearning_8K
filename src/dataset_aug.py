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
)
from preprocess import preprocess_audio_file


CLASS_TO_INDEX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_INDEX.items()}


class SpectrogramAugmentation:
    def __init__(
        self,
        noise_prob=0.5,
        noise_std=0.015,
        time_shift_prob=0.5,
        max_time_shift=10,
        freq_mask_prob=0.7,
        max_freq_mask_width=12,
        num_freq_masks=2,
        time_mask_prob=0.7,
        max_time_mask_width=15,
        num_time_masks=2,
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

        self.df = pd.read_csv(self.metadata_path)

        self.df["audio_path"] = self.df.apply(
            lambda row: self.audio_dir / f"fold{int(row['fold'])}" / row["slice_file_name"],
            axis=1
        )

        self.df = self.df[self.df["audio_path"].apply(lambda x: x.exists())].reset_index(drop=True)
        self.df["label"] = self.df["class"].map(CLASS_TO_INDEX)

        if self.split == "train":
            if self.val_fold is None:
                raise ValueError("val_fold không được để None khi split='train'")
            self.df = self.df[
                (self.df["fold"] != self.test_fold) & (self.df["fold"] != self.val_fold)
            ].reset_index(drop=True)

        elif self.split == "val":
            if self.val_fold is None:
                raise ValueError("val_fold không được để None khi split='val'")
            self.df = self.df[self.df["fold"] == self.val_fold].reset_index(drop=True)

        elif self.split == "test":
            self.df = self.df[self.df["fold"] == self.test_fold].reset_index(drop=True)

        else:
            raise ValueError("split phải là 'train', 'val' hoặc 'test'")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = row["audio_path"]
        label = int(row["label"])

        feature = preprocess_audio_file(audio_path)

        if self.augment:
            feature = self.augmenter(feature)

        label = torch.tensor(label, dtype=torch.long)

        return {
            "feature": feature,
            "label": label,
            "class_name": row["class"],
            "file_name": row["slice_file_name"],
            "fold": int(row["fold"]),
            "audio_path": str(audio_path),
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
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def get_dataloaders(self):
        return self.get_train_loader(), self.get_val_loader(), self.get_test_loader()


if __name__ == "__main__":
    data_module = UrbanSoundDataLoaderAug()

    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    print("Train size:", len(data_module.train_dataset))
    print("Val size:", len(data_module.val_dataset))
    print("Test size:", len(data_module.test_dataset))

    batch = next(iter(train_loader))
    print("Batch feature shape:", batch["feature"].shape)
    print("Batch label shape:", batch["label"].shape)
    print("Batch class names:", batch["class_name"][:5])