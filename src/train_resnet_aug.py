from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import f1_score

from config import CLASS_NAMES, BATCH_SIZE
from dataset_aug import UrbanSoundDataLoaderAug
from resnet import ResNet18


NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
SEED = 42

EARLY_STOPPING_PATIENCE = 10
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5
BASE_CHANNELS = 64
GRAD_CLIP = 3.0

MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.5

WANDB_PROJECT = "urban-sound-classification"
WANDB_RUN_NAME = "resnet18_aug_full"
WANDB_GROUP = "resnet18_aug"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a = y
    y_b = y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        features = batch["feature"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        use_mixup = random.random() < MIXUP_PROB

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            if use_mixup:
                mixed_features, labels_a, labels_b, lam = mixup_data(features, labels, MIXUP_ALPHA)
                outputs = model(mixed_features)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = float((np.array(all_preds) == np.array(all_labels)).mean())
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


def save_history(history, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "train_acc",
            "train_macro_f1",
            "val_loss",
            "val_acc",
            "val_macro_f1",
        ])
        for row in history:
            writer.writerow([
                row["epoch"],
                row["lr"],
                row["train_loss"],
                row["train_acc"],
                row["train_macro_f1"],
                row["val_loss"],
                row["val_acc"],
                row["val_macro_f1"],
            ])


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_module = UrbanSoundDataLoaderAug()
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    model = ResNet18(
        num_classes=len(CLASS_NAMES),
        base_channels=BASE_CHANNELS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group=WANDB_GROUP,
        job_type="train",
        config={
            "model_name": "resnet18_small_mixup",
            "num_classes": len(CLASS_NAMES),
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "dropout": DROPOUT,
            "base_channels": BASE_CHANNELS,
            "label_smoothing": LABEL_SMOOTHING,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "grad_clip": GRAD_CLIP,
            "mixup_alpha": MIXUP_ALPHA,
            "mixup_prob": MIXUP_PROB,
        },
    )

    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("Device:", device)
    print("Train size:", len(data_module.train_dataset))
    print("Val size:", len(data_module.val_dataset))
    print("Test size:", len(data_module.test_dataset))

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )

        val_loss, val_acc, val_f1 = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
        })

        run.log({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1,
            "best_val_acc_so_far": max(best_val_acc, val_acc),
        })

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"LR:         {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f} | Val F1:   {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_path = checkpoint_dir / "resnet18_aug_best.pth"
            torch.save(model.state_dict(), best_model_path)
            run.summary["best_val_acc"] = best_val_acc
            run.summary["best_epoch"] = best_epoch
            run.summary["best_model_path"] = str(best_model_path)
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement count: {epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}")

        print("-" * 50)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    last_model_path = checkpoint_dir / "resnet18_aug_last.pth"
    history_path = log_dir / "resnet18_aug_history.csv"

    torch.save(model.state_dict(), last_model_path)
    save_history(history, history_path)

    run.summary["final_epoch"] = history[-1]["epoch"]
    run.summary["last_model_path"] = str(last_model_path)
    run.summary["history_path"] = str(history_path)
    run.finish()

    print("Training finished.")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best model path: {checkpoint_dir / 'resnet18_aug_best.pth'}")
    print(f"Last model path: {last_model_path}")
    print(f"History path: {history_path}")


if __name__ == "__main__":
    main()