import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0

        for _ in range(self.num_iterations):
            # y= w @ a + b
            linear_output = np.dot(inputs, self.weights) + self.intercept
            # 轉機率
            y_pred = self.sigmoid(linear_output)
            # 做更新
            dw = (1 / n_samples) * np.dot(inputs.T, (y_pred - targets))
            db = (1 / n_samples) * np.sum(y_pred - targets)

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, inputs) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        linear_output = np.dot(inputs, self.weights) + self.intercept
        y_pred_probs = self.sigmoid(linear_output)
        y_pred_classes = (y_pred_probs >= 0.5).astype(int)
        return y_pred_probs, y_pred_classes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class FLD:

    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
        # sequence，更通用
    ) -> None:

        x0 = inputs[np.array(targets) == 0]
        x1 = inputs[np.array(targets) == 1]
        self.m0 = np.mean(x0, axis=0)
        self.m1 = np.mean(x1, axis=0)

        # 類內
        s0 = np.dot((x0 - self.m0).T, (x0 - self.m0))
        s1 = np.dot((x1 - self.m1).T, (x1 - self.m1))
        self.sw = s0 + s1
        # 類間
        mean_diff = (self.m1 - self.m0).reshape(-1, 1)
        self.sb = np.dot(mean_diff, mean_diff.T)

        # FLD投影
        self.w = np.dot(np.linalg.inv(self.sw), (self.m1 - self.m0))
        self.slope = -self.w[0] / self.w[1] if self.w[1] != 0 else None

    def predict(self, inputs: npt.NDArray[float]) -> t.Sequence[t.Union[int, bool]]:
        projections = np.dot(inputs, self.w)
        m0_proj = np.dot(self.m0, self.w)
        m1_proj = np.dot(self.m1, self.w)
        # 取中間為threshold
        threshold = (m0_proj + m1_proj) / 2
        preds = (projections > threshold).astype(int)
        return preds

    def plot_projection(self, inputs: npt.NDArray[float], targets: t.Sequence[int]):
        projections = np.dot(inputs, self.w)
        m0_proj = np.dot(self.m0, self.w)
        m1_proj = np.dot(self.m1, self.w)
        threshold = (m0_proj + m1_proj) / 2

        preds = (projections > threshold).astype(int)
        acc = np.mean(preds == targets)

        plt.figure(figsize=(6, 6))
        plt.title(
            f"Projection onto FLD axis (Acc={acc:.3f})\n"
            f"w=({self.w[0]:.3f}, {self.w[1]:.3f})"
        )

        for i in range(len(inputs)):
            color = 'green' if preds[i] == targets[i] else 'red'
            marker = 'o' if targets[i] == 0 else '^'
            plt.scatter(inputs[i, 0], inputs[i, 1], c=color, marker=marker, s=40, alpha=0.7)

        # 投影線 (灰色)
        slope = self.w[1] / self.w[0] if self.w[0] != 0 else np.inf
        x_vals = np.linspace(inputs[:, 0].min(), inputs[:, 0].max(), 100)
        y_vals = slope * x_vals
        plt.plot(x_vals, y_vals, color='gray', label='Projection line')

        # 決策線 (藍色)
        if self.w[1] != 0:
            y_boundary = -(self.w[0] / self.w[1]) * x_vals + threshold / self.w[1]
            plt.plot(x_vals, y_boundary, color='blue', linestyle='--', label='Decision boundary')

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


def compute_auc(y_trues, y_preds):
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    sorted_idx = np.argsort(y_preds)
    y_trues = y_trues[sorted_idx]
    y_preds = y_preds[sorted_idx]

    tpr = []
    fpr = []
    P = np.sum(y_trues == 1)
    N = np.sum(y_trues == 0)

    thresholds = np.unique(y_preds)
    for th in thresholds:
        y_hat = (y_preds >= th).astype(int)
        TP = np.sum((y_hat == 1) & (y_trues == 1))
        FP = np.sum((y_hat == 1) & (y_trues == 0))
        TPR = TP / P if P else 0
        FPR = FP / N if N else 0
        tpr.append(TPR)
        fpr.append(FPR)
    # 用梯形法算面積
    auc = abs(np.trapz(tpr, fpr))
    return auc


def accuracy_score(y_trues, y_preds):
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    return np.mean(y_trues == y_preds)


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-2,  # You can modify the parameters as you want
        num_iterations=30000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    FLD_.plot_projection(x_test, y_test)


if __name__ == '__main__':
    main()
