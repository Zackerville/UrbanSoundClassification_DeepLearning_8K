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


class UrbanSoundDataset(Dataset):
    def __init__(self, split, test_fold, val_fold=None, metadata_path=METADATA_PATH, audio_dir=AUDIO_DIR):
        self.split = split
        self.test_fold = int(test_fold)
        self.val_fold = None if val_fold is None else int(val_fold)
        self.metadata_path = metadata_path
        self.audio_dir = audio_dir

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
            return {
                "feature": feature,
                "label": self._labels[idx].clone(),
                "class_name": self._class_names[idx],
                "file_name": self._file_names[idx],
                "fold": int(self._folds[idx]),
                "audio_path": "",
            }

        row = self.df.iloc[idx]
        feature = preprocess_audio_file(row["audio_path"])
        return {
            "feature": feature,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "class_name": row["class"],
            "file_name": row["slice_file_name"],
            "fold": int(row["fold"]),
            "audio_path": str(row["audio_path"]),
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
    data_module = UrbanSoundDataLoader()

    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    print("Cache mode:", data_module.train_dataset.use_cache)
    print("Train size:", len(data_module.train_dataset))
    print("Val size:", len(data_module.val_dataset))
    print("Test size:", len(data_module.test_dataset))

    batch = next(iter(train_loader))
    print("Batch feature shape:", batch["feature"].shape)
    print("Batch label shape:", batch["label"].shape)
    print("Batch class names:", batch["class_name"][:5])
