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
from dataset_aug import UrbanSoundDataLoaderAug
from resnet import ResNet18


WANDB_PROJECT = "urban-sound-classification"
WANDB_RUN_NAME = "ensemble_resnet_reg_aug_tta_eval"
WANDB_GROUP = "ensemble"

RESNET_REG_TAG = "resnet18_regular_v3"
RESNET_REG_DEFAULT_BASE_CHANNELS = 32
RESNET_REG_DEFAULT_DROPOUT = 0.3

RESNET_AUG_BASE_CHANNELS = 64
RESNET_AUG_DROPOUT = 0.5

TTA_SHIFT = 4

ENSEMBLE_WEIGHT_REG = 0.5
ENSEMBLE_WEIGHT_AUG = 0.5


def forward_tta(model, features):
    outputs_1 = model(features)
    outputs_2 = model(torch.roll(features, shifts=TTA_SHIFT, dims=-1))
    outputs_3 = model(torch.roll(features, shifts=-TTA_SHIFT, dims=-1))
    return (outputs_1 + outputs_2 + outputs_3) / 3.0


def load_resnet_regular(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        base_channels = checkpoint.get("base_channels", RESNET_REG_DEFAULT_BASE_CHANNELS)
        dropout = checkpoint.get("dropout", RESNET_REG_DEFAULT_DROPOUT)
    else:
        state_dict = checkpoint
        base_channels = RESNET_REG_DEFAULT_BASE_CHANNELS
        dropout = RESNET_REG_DEFAULT_DROPOUT

    model = ResNet18(
        num_classes=len(CLASS_NAMES),
        base_channels=base_channels,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(state_dict)
    return model, base_channels, dropout


def load_resnet_aug(checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    model = ResNet18(
        num_classes=len(CLASS_NAMES),
        base_channels=RESNET_AUG_BASE_CHANNELS,
        dropout=RESNET_AUG_DROPOUT,
    ).to(device)
    model.load_state_dict(state_dict)
    return model


def evaluate_ensemble(model_reg, model_aug, loader, criterion, device):
    model_reg.eval()
    model_aug.eval()

    running_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []
    all_file_names = []
    all_class_names = []

    with torch.no_grad():
        for batch in loader:
            features = batch["feature"].to(device)
            labels = batch["label"].to(device)

            logits_reg = forward_tta(model_reg, features)
            logits_aug = forward_tta(model_aug, features)

            probs_reg = torch.softmax(logits_reg, dim=1)
            probs_aug = torch.softmax(logits_aug, dim=1)

            probs = (
                ENSEMBLE_WEIGHT_REG * probs_reg
                + ENSEMBLE_WEIGHT_AUG * probs_aug
            )

            loss = criterion(torch.log(probs + 1e-8), labels)
            preds = probs.argmax(dim=1)

            running_loss += loss.item() * features.size(0)
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_file_names.extend(batch["file_name"])
            all_class_names.extend(batch["class_name"])

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
        "class_names": all_class_names,
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


def save_predictions(labels, preds, probs, file_names, class_names, save_path):
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
            file_names, class_names, labels, preds, probs
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
    checkpoint_reg_path = root_dir / "outputs" / "checkpoints" / f"{RESNET_REG_TAG}_best.pth"
    checkpoint_aug_path = root_dir / "outputs" / "checkpoints" / "resnet18_aug_best.pth"
    evaluation_dir = root_dir / "outputs" / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_reg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ResNet regular checkpoint: {checkpoint_reg_path}")
    if not checkpoint_aug_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ResNet aug checkpoint: {checkpoint_aug_path}")

    data_module = UrbanSoundDataLoaderAug(num_workers=0)
    _, _, test_loader = data_module.get_dataloaders()

    model_reg, reg_base_channels, reg_dropout = load_resnet_regular(checkpoint_reg_path, device)
    model_aug = load_resnet_aug(checkpoint_aug_path, device)

    criterion = nn.NLLLoss()

    metrics = evaluate_ensemble(
        model_reg=model_reg,
        model_aug=model_aug,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    metrics_path = evaluation_dir / "ensemble_test_metrics_tta.json"
    report_path = evaluation_dir / "ensemble_classification_report_tta.csv"
    predictions_path = evaluation_dir / "ensemble_test_predictions_tta.csv"
    cm_csv_path = evaluation_dir / "ensemble_confusion_matrix_tta.csv"
    cm_img_path = evaluation_dir / "ensemble_confusion_matrix_tta.png"

    save_metrics(metrics, metrics_path)
    save_classification_report(metrics["report_dict"], report_path)
    save_predictions(
        metrics["labels"],
        metrics["preds"],
        metrics["probs"],
        metrics["file_names"],
        metrics["class_names"],
        predictions_path,
    )
    save_confusion_matrix_csv(metrics["confusion_matrix"], cm_csv_path)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        CLASS_NAMES,
        cm_img_path,
        "Ensemble (ResNet18 + ResNet18-aug) TTA Confusion Matrix",
    )

    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        group=WANDB_GROUP,
        job_type="evaluate",
        config={
            "model_name": "ensemble_resnet_reg_aug",
            "checkpoint_reg_path": str(checkpoint_reg_path),
            "checkpoint_aug_path": str(checkpoint_aug_path),
            "num_classes": len(CLASS_NAMES),
            "reg_base_channels": reg_base_channels,
            "reg_dropout": reg_dropout,
            "aug_base_channels": RESNET_AUG_BASE_CHANNELS,
            "aug_dropout": RESNET_AUG_DROPOUT,
            "ensemble_weight_reg": ENSEMBLE_WEIGHT_REG,
            "ensemble_weight_aug": ENSEMBLE_WEIGHT_AUG,
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

    print("Ensemble evaluation finished.")
    print(f"ResNet regular checkpoint: {checkpoint_reg_path}")
    print(f"ResNet aug checkpoint:     {checkpoint_aug_path}")
    print(f"Ensemble weights: reg={ENSEMBLE_WEIGHT_REG}, aug={ENSEMBLE_WEIGHT_AUG}")
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
