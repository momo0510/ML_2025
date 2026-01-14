import numpy as np
import pandas as pd
from loguru import logger
import random
import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, plot_feature_importance


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Simple standardization (z-score) using training statistics."""
    train_x = train_df.drop(["target"], axis=1)
    test_x = test_df.drop(["target"], axis=1)

    # 將所有類別文字轉成 one-hot 編碼（保持欄位一致）
    full = pd.concat([train_x, test_x], axis=0)
    full_encoded = pd.get_dummies(full, drop_first=True)  # 轉為數值
    x_train = full_encoded.iloc[: len(train_x), :].to_numpy(dtype=np.float32)
    x_test = full_encoded.iloc[len(train_x):, :].to_numpy(dtype=np.float32)

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-8

    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, x_test


def accuracy_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def main():
    """You can control the seed for reproducibility"""
    random.seed(777)
    torch.manual_seed(777)

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    # X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    feature_names = list(train_df.drop(['target'], axis=1).columns)
    logger.info(f"Num features: {len(feature_names)}")

    """
    TODO: Implement you preprocessing function.
    """
    X_train, X_test = preprocess(train_df, test_df)

    input_dim = X_train.shape[1]
    """
    TODO: Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    logger.info("Training AdaBoost ...")
    clf_adaboost = AdaBoostClassifier(input_dim=input_dim, num_learners=10)
    clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=1000,
        learning_rate=0.005,
    )

    ada_pred_classes, ada_learners_probs = clf_adaboost.predict_learners(X_test)
    ada_acc = accuracy_score(y_test, ada_pred_classes)
    logger.info(f"AdaBoost - Accuracy: {ada_acc:.4f}")
    plot_learners_roc(
        y_preds=ada_learners_probs,
        y_trues=y_test,
        fpath="./adaboost_roc.png",
    )
    ada_feature_importance = clf_adaboost.compute_feature_importance()
    plot_feature_importance(
        feature_names,
        ada_feature_importance,
        fpath="./adaboost_feature_importance.png"
    )

    # Bagging
    logger.info("Training Bagging ...")
    clf_bagging = BaggingClassifier(input_dim=input_dim)
    clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.01,
    )

    bag_pred_classes, bag_learners_probs = clf_bagging.predict_learners(X_test)
    bag_acc = accuracy_score(y_test, bag_pred_classes)
    logger.info(f"Bagging - Accuracy: {bag_acc:.4f}")
    plot_learners_roc(
        y_preds=bag_learners_probs,
        y_trues=y_test,
        fpath="./bagging_roc.png",
    )
    bag_feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(
        feature_names,
        bag_feature_importance,
        fpath="./bagging_feature_importance.png"
    )

    # Decision Tree
    logger.info("Training Decision Tree ...")
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    tree_pred_classes = clf_tree.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_pred_classes)
    logger.info(f"DecisionTree - Accuracy: {tree_acc:.4f}")

    plot_feature_importance(
        feature_names,
        clf_tree.feature_importances_,
        fpath="./decisiontree_feature_importance.png"
    )


if __name__ == '__main__':
    main()
