import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas: t.List[float] = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, num_epochs: int = 2000, learning_rate: float = 0.002):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        n_samples = X_train.shape[0]

        self.sample_weights = np.ones(n_samples, dtype=np.float64) / n_samples

        x_tensor_full = torch.from_numpy(X_train)

        for t, learner in enumerate(self.learners):
            learner.train() # 開啟訓練模式代表可以更新梯度，但可以不開不影響
            y_tensor_full = torch.from_numpy(y_train)

            optimizer = optim.SGD(learner.parameters(), lr=learning_rate) # 管理參數用，在每次 optimizer.step() 時根據梯度更新權重。
            for _ in range(num_epochs):
                optimizer.zero_grad() # 清空梯度
                outputs = learner(x_tensor_full)  # logits，模型的線性輸出
                losses = entropy_loss(outputs, y_tensor_full)  

                w_tensor = torch.from_numpy(self.sample_weights).float()
                loss = torch.sum(losses * w_tensor) / torch.sum(w_tensor) # 把誤差和樣本銓重相乘
                loss.backward() # 算梯度(learner.linear.weight.grad 和 learner.linear.bias.grad )  
                optimizer.step() #更新模型權重

            # compute weighted error
            with torch.no_grad():
                outputs = learner(x_tensor_full)
                probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(np.float32)

            misclassified = (preds != y_train).astype(np.float64)
            error_t = float(np.sum(self.sample_weights * misclassified))

            # avoid division by zero
            error_t = np.clip(error_t, 1e-10, 1 - 1e-10)

            alpha_t = 0.5 * np.log((1.0 - error_t) / error_t)
            self.alphas.append(alpha_t)

            # update sample weights: w_i <- w_i * exp(-alpha_t * y_i * h_i)
            # convert labels from {0,1} to {-1,+1}
            y_pm = 2 * y_train - 1
            h_pm = 2 * preds - 1
            self.sample_weights *= np.exp(-alpha_t * y_pm * h_pm)
            self.sample_weights /= np.sum(self.sample_weights)

        return self

    def predict_learners(self, X: np.ndarray,) -> t.Tuple[t.Sequence[int], t.List[t.Sequence[float]]]:
        X = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(X)

        learners_probs: t.List[np.ndarray] = []
        learners_votes: t.List[np.ndarray] = []

        for learner in self.learners:
            learner.eval() # 推論模式
            with torch.no_grad():
                outputs = learner(x_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
            learners_probs.append(probs)
            votes = (probs >= 0.5).astype(np.float32)  # {0,1}
            learners_votes.append(2 * votes - 1)  # {-1,+1}

        # aggregate with alpha weights
        F = np.zeros_like(learners_votes[0])
        for alpha_t, h_t in zip(self.alphas, learners_votes):
            F += alpha_t * h_t

        final_preds = (F >= 0).astype(int)  # back to {0,1}
        return final_preds, learners_probs


    def compute_feature_importance(self) -> t.Sequence[float]:
        if not self.learners:
            return []

        first_weight = next(self.learners[0].parameters()).detach().cpu().numpy()
        n_features = first_weight.shape[-1]

        importance = np.zeros(n_features, dtype=np.float64)

        for alpha_t, learner in zip(self.alphas, self.learners):
            w = learner.linear.weight.detach().cpu().numpy().reshape(-1)
            importance += abs(alpha_t) * np.abs(w)

        if importance.sum() > 0:
            importance = importance / importance.sum()           
        return importance
