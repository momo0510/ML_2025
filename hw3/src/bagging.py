import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, num_epochs: int = 1000, learning_rate: float = 0.1):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        n_samples = X_train.shape[0]

        for learner in self.learners:
            # bootstrap sampling (取後放回，取次數=size)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]

            x_tensor = torch.from_numpy(X_boot)
            y_tensor = torch.from_numpy(y_boot)

            learner.train()
            optimizer = optim.SGD(learner.parameters(), lr=learning_rate)

            for _ in range(num_epochs):
                optimizer.zero_grad()
                outputs = learner(x_tensor)
                losses = entropy_loss(outputs, y_tensor)
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

        return self

    def predict_learners(self, X: np.ndarray,) -> t.Tuple[t.Sequence[int], t.List[t.Sequence[float]]]:
        X = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(X)

        learners_probs: t.List[np.ndarray] = []

        for learner in self.learners:
            learner.eval()
            with torch.no_grad():
                outputs = learner(x_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
            learners_probs.append(probs)

        avg_probs = np.mean(np.stack(learners_probs, axis=0), axis=0)
        final_preds = (avg_probs >= 0.5).astype(int)

        return final_preds, learners_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        if not self.learners:
            return []

        first_weight = next(self.learners[0].parameters()).detach().cpu().numpy()
        n_features = first_weight.shape[-1]

        importance = np.zeros(n_features, dtype=np.float64)

        for learner in self.learners:
            w = learner.linear.weight.detach().cpu().numpy().reshape(-1)
            importance += np.abs(w)

        importance /= len(self.learners)
        if importance.sum() > 0:
            importance = importance / importance.sum()
        return importance
