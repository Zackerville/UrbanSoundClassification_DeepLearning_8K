from pathlib import Path
import csv
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from config import CLASS_NAMES, BATCH_SIZE
from dataset import UrbanSoundDataLoader
from cnn_baseline import CNNBaseline


NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LR = 1e-6

EARLY_STOPPING_PATIENCE = 5

WANDB_PROJECT = "urban-sound-classification"
WANDB_RUN_NAME = "cnn_baseline"
WANDB_GROUP = "cnn_baseline"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        features = batch["feature"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_history(history, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])
        for row in history:
            writer.writerow([
                row["epoch"],
                row["lr"],
                row["train_loss"],
                row["train_acc"],
                row["val_loss"],
                row["val_acc"],
            ])


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_module = UrbanSoundDataLoader()
    train_loader, val_loader, test_loader = data_module.get_dataloaders()

    model = CNNBaseline(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LR,
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group=WANDB_GROUP,
        job_type="train",
        config={
            "model_name": "cnn_baseline",
            "num_classes": len(CLASS_NAMES),
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "scheduler_factor": SCHEDULER_FACTOR,
            "scheduler_patience": SCHEDULER_PATIENCE,
            "min_lr": MIN_LR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
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

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(val_loss)

        history.append({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        run.log({
            "epoch": epoch + 1,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc_so_far": max(best_val_acc, val_acc),
        })

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")
        print(f"LR:         {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_path = checkpoint_dir / "cnn_baseline_best.pth"
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

    last_model_path = checkpoint_dir / "cnn_baseline_last.pth"
    history_path = log_dir / "cnn_baseline_history.csv"

    torch.save(model.state_dict(), last_model_path)
    save_history(history, history_path)

    run.summary["final_epoch"] = history[-1]["epoch"]
    run.summary["last_model_path"] = str(last_model_path)
    run.summary["history_path"] = str(history_path)
    run.finish()

    print("Training finished.")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best model path: {checkpoint_dir / 'cnn_baseline_best.pth'}")
    print(f"Last model path: {last_model_path}")
    print(f"History path: {history_path}")


if __name__ == "__main__":
    main()