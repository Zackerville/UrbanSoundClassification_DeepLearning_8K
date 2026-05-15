from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config import CLASS_NAMES
from dataset import UrbanSoundDataLoader
from resnet import ResNet18


RUN_TAG = "resnet18_regular_b32_d05_mix06"
WANDB_PROJECT = "urban-sound-classification"
WANDB_RUN_NAME = f"{RUN_TAG}_tta_eval"
WANDB_GROUP = "resnet18_regular"

DEFAULT_BASE_CHANNELS = 32
DEFAULT_DROPOUT = 0.5
TTA_SHIFT = 4


def forward_tta(model, features):
    outputs_1 = model(features)
    outputs_2 = model(torch.roll(features, shifts=TTA_SHIFT, dims=-1))
    outputs_3 = model(torch.roll(features, shifts=-TTA_SHIFT, dims=-1))
    return (outputs_1 + outputs_2 + outputs_3) / 3.0


def evaluate_model(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []
    all_file_names = []
    all_true_class_names = []

    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            outputs = forward_tta(model, features)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * features.size(0)
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_file_names.extend(batch["file_name"])
            all_true_class_names.extend(batch["class_name"])

    test_loss = running_loss / total
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)

    report_dict = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    report_text = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    return {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "report_dict": report_dict,
        "report_text": report_text,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
        "file_names": all_file_names,
        "true_class_names": all_true_class_names,
    }


def save_metrics(metrics, save_path):
    data = {
        "test_loss": metrics["test_loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_classification_report(report_dict, save_path):
    rows = []

    for label, values in report_dict.items():
        if isinstance(values, dict):
            row = {"label": label}
            row.update(values)
            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_predictions(labels, preds, probs, file_names, true_class_names, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name",
            "true_class_name",
            "true_label",
            "pred_label",
            "pred_class_name",
            "pred_confidence",
        ])

        for file_name, true_class_name, true_label, pred_label, prob_vector in zip(
            file_names, true_class_names, labels, preds, probs
        ):
            pred_confidence = float(prob_vector[pred_label])
            writer.writerow([
                file_name,
                true_class_name,
                true_label,
                pred_label,
                CLASS_NAMES[pred_label],
                pred_confidence,
            ])


def save_confusion_matrix_csv(cm, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + CLASS_NAMES)

        for class_name, row in zip(CLASS_NAMES, cm):
            writer.writerow([class_name] + row.tolist())


def plot_confusion_matrix(cm, class_names, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = Path(__file__).resolve().parent.parent
    checkpoint_path = root_dir / "outputs" / "checkpoints" / f"{RUN_TAG}_best.pth"
    evaluation_dir = root_dir / "outputs" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        base_channels = checkpoint.get("base_channels", DEFAULT_BASE_CHANNELS)
        dropout = checkpoint.get("dropout", DEFAULT_DROPOUT)
    else:
        state_dict = checkpoint
        base_channels = DEFAULT_BASE_CHANNELS
        dropout = DEFAULT_DROPOUT

    data_module = UrbanSoundDataLoader(batch_size=None if False else None)
    data_module = UrbanSoundDataLoader(num_workers=0)
    _, _, test_loader = data_module.get_dataloaders()

    model = ResNet18(
        num_classes=len(CLASS_NAMES),
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    metrics = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics_path = evaluation_dir / f"{RUN_TAG}_test_metrics_tta.json"
    report_path = evaluation_dir / f"{RUN_TAG}_classification_report_tta.csv"
    predictions_path = evaluation_dir / f"{RUN_TAG}_test_predictions_tta.csv"
    cm_csv_path = evaluation_dir / f"{RUN_TAG}_confusion_matrix_tta.csv"
    cm_img_path = evaluation_dir / f"{RUN_TAG}_confusion_matrix_tta.png"

    save_metrics(metrics, metrics_path)
    save_classification_report(metrics["report_dict"], report_path)
    save_predictions(
        metrics["labels"],
        metrics["preds"],
        metrics["probs"],
        metrics["file_names"],
        metrics["true_class_names"],
        predictions_path,
    )
    save_confusion_matrix_csv(metrics["confusion_matrix"], cm_csv_path)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CLASS_NAMES,
        cm_img_path,
        f"{RUN_TAG} TTA Confusion Matrix",
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group=WANDB_GROUP,
        job_type="evaluate",
        config={
            "run_tag": RUN_TAG,
            "checkpoint_path": str(checkpoint_path),
            "num_classes": len(CLASS_NAMES),
            "base_channels": base_channels,
            "dropout": dropout,
            "tta_shift": TTA_SHIFT,
        },
    )

    run.log({
        "test_loss": metrics["test_loss"],
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=metrics["labels"],
            preds=metrics["preds"],
            class_names=CLASS_NAMES,
        ),
        "confusion_matrix_image": wandb.Image(str(cm_img_path)),
    })

    run.summary["metrics_path"] = str(metrics_path)
    run.summary["report_path"] = str(report_path)
    run.summary["predictions_path"] = str(predictions_path)
    run.summary["confusion_matrix_csv_path"] = str(cm_csv_path)
    run.summary["confusion_matrix_image_path"] = str(cm_img_path)
    run.finish()

    print("Evaluation finished.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {metrics['macro_f1']:.4f}")
    print()
    print(metrics["report_text"])
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved report to: {report_path}")
    print(f"Saved predictions to: {predictions_path}")
    print(f"Saved confusion matrix csv to: {cm_csv_path}")
    print(f"Saved confusion matrix image to: {cm_img_path}")


if __name__ == "__main__":
    main()