"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
import os


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # Closed-form
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        w_all = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
        self.intercept = w_all[0]
        self.weights = w_all[1:]
        return self

    def predict(self, X):
        return X @ self.weights + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate=1e-2, epochs=2000, decay=1e-4):
        y = y.reshape(-1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0.0

        losses, lr_history = [], []
        for epoch in range(epochs):
            y_pred = X @ self.weights + self.intercept
            error = y_pred - y

            loss = np.mean(error ** 2)
            # 畫線
            losses.append(loss)

            # decay學習綠
            lr = learning_rate / (1 + decay * epoch)
            lr_history.append(lr)

            # 做偏微和更新
            dw = (2 / n_samples) * X.T @ error
            db = (2 / n_samples) * np.sum(error)
            self.weights -= lr * dw
            self.intercept -= lr * db

            if epoch % 100 == 0:
                logger.info(f"EPOCH {epoch}, loss={loss:.4f}, lr={lr:.6f}")

        return losses, lr_history

    def predict(self, X):
        return X @ self.weights + self.intercept


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    current_dir = os.path.dirname(__file__)
    train_df = pd.read_csv(os.path.join(current_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(current_dir, 'test.csv'))

    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    # Normalize features
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)

    """q1"""
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    # Gradient Descent solution 
    LR_GD = LinearRegressionGradientdescent()
    learning_rate = 1e-2
    epochs = 2000
    decay = 1e-4
    losses ,lr_history = LR_GD.fit(train_x, train_y,
                                   learning_rate=learning_rate,
                                   epochs=epochs, decay=decay)

    """q2"""
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    """q3"""
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss (MSE)')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(current_dir, 'learning_curve.png'))
    plt.show()

    """q4"""
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
