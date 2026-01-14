import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features, dtype=float)
        self.tree = self._grow_tree(X, y, depth=0)
        # normalize feature importances
        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (
            depth >= self.max_depth
            or num_labels == 1
            or num_samples < 2
        ):
            # majority vote
            values, counts = np.unique(y, return_counts=True)
            leaf_label = values[np.argmax(counts)]
            return {"type": "leaf", "class": int(leaf_label)}

        # find best split
        feature_idx, threshold, gain = find_best_split(X, y)
        if feature_idx is None:
            values, counts = np.unique(y, return_counts=True)
            leaf_label = values[np.argmax(counts)]
            return {"type": "leaf", "class": int(leaf_label)}

        # accumulate feature importance by information gain
        self.feature_importances_[feature_idx] += gain

        X_left, X_right, y_left, y_right = split_dataset(
            X, y, feature_idx, threshold
        )

        left_subtree = self._grow_tree(X_left, y_left, depth + 1)
        right_subtree = self._grow_tree(X_right, y_right, depth + 1)

        return {
            "type": "node",
            "feature_index": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def predict(self, X: np.ndarray):
        X = np.asarray(X)
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        if tree_node["type"] == "leaf":
            return tree_node["class"]
        feature_idx = tree_node["feature_index"]
        threshold = tree_node["threshold"]
        if x[feature_idx] <= threshold:
            return self._predict_tree(x, tree_node["left"])
        return self._predict_tree(x, tree_node["right"])


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    X = np.asarray(X)
    y = np.asarray(y)
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]


# Find the best split for the dataset
def find_best_split(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    num_samples, num_features = X.shape

    if num_samples <= 1:
        return None, None, 0.0

    parent_entropy = entropy(y)
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for feature_idx in range(num_features):
        values = np.unique(X[:, feature_idx])
        for thresh in values:
            X_left, X_right, y_left, y_right = split_dataset(
                X, y, feature_idx, thresh
            )
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            p_left = float(len(y_left)) / num_samples
            p_right = 1.0 - p_left
            gain = parent_entropy - (
                p_left * entropy(y_left) + p_right * entropy(y_right)
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = thresh

    return best_feature, best_threshold, best_gain


def entropy(y):
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))