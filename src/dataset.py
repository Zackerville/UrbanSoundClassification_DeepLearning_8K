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


class UrbanSoundDataset(Dataset):
    def __init__(self, split, test_fold, val_fold=None, metadata_path=METADATA_PATH, audio_dir=AUDIO_DIR):
        self.split = split
        self.test_fold = int(test_fold)
        self.val_fold = None if val_fold is None else int(val_fold)
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir

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
        label = torch.tensor(label, dtype=torch.long)

        return {
            "feature": feature,
            "label": label,
            "class_name": row["class"],
            "file_name": row["slice_file_name"],
            "fold": int(row["fold"]),
            "audio_path": str(audio_path),
        }


class UrbanSoundDataLoader:
    def __init__(
        self,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_fold=TEST_FOLD,
        val_fold=VAL_FOLD,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_fold = test_fold
        self.val_fold = val_fold

        self.train_dataset = UrbanSoundDataset(
            split="train",
            test_fold=self.test_fold,
            val_fold=self.val_fold,
        )
        self.val_dataset = UrbanSoundDataset(
            split="val",
            test_fold=self.test_fold,
            val_fold=self.val_fold,
        )
        self.test_dataset = UrbanSoundDataset(
            split="test",
            test_fold=self.test_fold,
            val_fold=self.val_fold,
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
    data_module = UrbanSoundDataLoader()

    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    print("Train size:", len(data_module.train_dataset))
    print("Val size:", len(data_module.val_dataset))
    print("Test size:", len(data_module.test_dataset))

    batch = next(iter(train_loader))
    print("Batch feature shape:", batch["feature"].shape)
    print("Batch label shape:", batch["label"].shape)
    print("Batch class names:", batch["class_name"][:5])