import typing as t
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation in the `intermediate layers` allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_dim)
        out = self.linear(x)  # (batch_size, 1)
        return out.squeeze(1)  # (batch_size,)


def entropy_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy loss (per-sample, no reduction).
    outputs: logits, shape (N,)
    targets: {0,1}, shape (N,)
    """
    targets = targets.float()
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    return criterion(outputs, targets)


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath: str = "./tmp.png",
):
    """
    Plot ROC curve for each learner.
    y_preds: list of length T, each element is (N,) probabilities for learner t.
    y_trues: length N, ground-truth labels {0,1}.
    """
    y_trues = np.asarray(y_trues)

    plt.figure(figsize=(6, 6))
    for idx, preds in enumerate(y_preds):
        preds = np.asarray(preds)
        fpr, tpr, _ = roc_curve(y_trues, preds)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Learner {idx + 1} (AUC={auc_score:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves of Individual Learners")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


def plot_feature_importance(feature_names, importances, fpath="feature_importance.png"):
    """
    Plot feature importances without sorting.
    Feature count must match; if not, auto-adjust.
    """
    feature_names = np.array(feature_names)
    importances = np.array(importances)

    # If lengths mismatch: auto-fix
    min_len = min(len(feature_names), len(importances))
    feature_names = feature_names[:min_len]
    importances = importances[:min_len]

    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()