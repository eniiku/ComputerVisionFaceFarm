import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

FONT_SIZE = 12
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
LINE_WIDTH = 2
DPI = 100


def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss over epochs.
    Args:
        history (dict): The history dictionary from model.fit().
    """
    plt.style.use("seaborn-v0_8-darkgrid")

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    epochs_range = range(len(acc)) if acc else range(len(loss))

    plt.figure(figsize=(14, 6), dpi=DPI)

    plt.subplot(1, 2, 1)
    if acc and val_acc:
        plt.plot(epochs_range, acc, label="Training Accuracy", linewidth=LINE_WIDTH)
        plt.plot(
            epochs_range, val_acc, label="Validation Accuracy", linewidth=LINE_WIDTH
        )
    plt.legend(loc="lower right", fontsize=FONT_SIZE)
    plt.title("Training and Validation Accuracy", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Accuracy", fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    if loss and val_loss:
        plt.plot(epochs_range, loss, label="Training Loss", linewidth=LINE_WIDTH)
        plt.plot(epochs_range, val_loss, label="Validation Loss", linewidth=LINE_WIDTH)
    plt.legend(loc="upper right", fontsize=FONT_SIZE)
    plt.title("Training and Validation Loss", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Epoch", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Loss", fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred_classes, class_names):
    """
    Plots the confusion matrix as a heatmap.
    """
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6), dpi=DPI)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="black",
        annot_kws={"size": FONT_SIZE * 1.2},
    )
    plt.title("Confusion Matrix", fontsize=TITLE_FONT_SIZE)
    plt.xlabel("Predicted Label", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("True Label", fontsize=LABEL_FONT_SIZE)
    plt.xticks(fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE, rotation=0)
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred_classes, class_names):
    """
    Prints the classification report.
    """
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred_classes, target_names=class_names))


def plot_roc_curve(y_true, y_pred_probs, class_names):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    """
    if len(np.unique(y_true)) < 2:
        print(
            "ROC Curve cannot be plotted: Requires at least two classes in true labels."
        )
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 7), dpi=DPI)
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=LINE_WIDTH,
        label=f"ROC curve (area = {roc_auc:.2f})",
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=LINE_WIDTH, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("True Positive Rate", fontsize=LABEL_FONT_SIZE)
    plt.title(
        f"Receiver Operating Characteristic (ROC) Curve - {class_names[1]}",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.legend(loc="lower right", fontsize=FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_probs, class_names):
    """
    Plots the Precision-Recall curve.
    """
    if len(np.unique(y_true)) < 2:
        print(
            "Precision-Recall Curve cannot be plotted: Requires at least two classes in true labels."
        )
        return

    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)

    plt.figure(figsize=(7, 7), dpi=DPI)
    plt.plot(
        recall,
        precision,
        color="b",
        lw=LINE_WIDTH,
        label=f"Precision-Recall curve (AP = {ap:.2f})",
    )
    plt.xlabel("Recall", fontsize=LABEL_FONT_SIZE)
    plt.ylabel("Precision", fontsize=LABEL_FONT_SIZE)
    plt.title(f"Precision-Recall Curve - {class_names[1]}", fontsize=TITLE_FONT_SIZE)
    plt.legend(loc="lower left", fontsize=FONT_SIZE)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
