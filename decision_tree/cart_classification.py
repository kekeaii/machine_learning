from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


def gini(labels: np.ndarray) -> float:
    """基尼指数 Gini(D)。"""
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(1.0 - np.sum(probs**2))


def majority_label(labels: np.ndarray) -> Any:
    """返回多数类，同时用于节点分类误差统计。"""
    values, counts = np.unique(labels, return_counts=True)
    value = values[np.argmax(counts)]
    return value.item() if hasattr(value, "item") else value


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    读数据:
    - CSV: pandas 读取，最后一列默认是标签 y
    - JSON: 支持 {"X": ..., "y": ...} 或 [{"x": ..., "y": ...}, ...]
    返回值统一转成 numpy 数组。
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least one feature column and one label column")
        feature_names = df.columns[:-1].tolist()
        X = df.iloc[:, :-1].to_numpy(dtype=float)
        y = df.iloc[:, -1].to_numpy(dtype=object)
        return X, y, feature_names

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "X" in data and "y" in data:
        X = np.asarray(data["X"], dtype=float)
        y = np.asarray(data["y"], dtype=object)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        return X, y, feature_names

    if isinstance(data, list):
        X = []
        y = []
        for item in data:
            if isinstance(item, dict) and "x" in item and "y" in item:
                X.append(item["x"])
                y.append(item["y"])
            elif isinstance(item, list) and len(item) == 2:
                X.append(item[0])
                y.append(item[1])
            else:
                raise ValueError("unsupported JSON item format")
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=object)
        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
        return X_arr, y_arr, feature_names

    raise ValueError("unsupported dataset format")


@dataclass
class CartNode:
    is_leaf: bool
    prediction: Any
    sample_count: int
    majority_count: int
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional["CartNode"] = None
    right: Optional["CartNode"] = None


class CARTClassifier:
    """
    手写 CART 分类树。
    - 划分指标: 基尼指数
    - 剪枝: CART 代价复杂度剪枝
    """

    def __init__(self, min_samples_split: int = 2, max_depth: Optional[int] = None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tree: Optional[CartNode] = None
        self.feature_names: Optional[list[str]] = None
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> None:
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")

        self._train_X = np.asarray(X, dtype=float)
        self._train_y = np.asarray(y, dtype=object)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self._train_X.shape[1])]
        self.tree = self._build_tree(self._train_X, self._train_y, depth=0)

    def predict_one(self, sample: np.ndarray) -> Any:
        if self.tree is None:
            raise ValueError("model is not fitted")
        node = self.tree
        while not node.is_leaf:
            if sample[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        return np.asarray([self.predict_one(row) for row in X_arr], dtype=object)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=object)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same length")
        if X_arr.shape[0] == 0:
            return 0.0
        predictions = self.predict(X_arr)
        return float(np.mean(predictions == y_arr))

    def prune(self, alpha: float = 0.0) -> None:
        """
        CART 代价复杂度剪枝:
        g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)
        若 g(t) <= alpha，则把子树 T_t 剪成叶子。
        这里 R 使用训练样本上的分类误差样本数。
        """
        if self.tree is None:
            raise ValueError("model is not fitted")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self._prune_node(self.tree, alpha)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> CartNode:
        prediction = majority_label(y)
        _, counts = np.unique(y, return_counts=True)
        majority_count = int(np.max(counts))
        node = CartNode(
            is_leaf=False,
            prediction=prediction,
            sample_count=int(y.size),
            majority_count=majority_count,
        )

        if np.unique(y).size == 1:
            node.is_leaf = True
            return node

        if y.size < self.min_samples_split:
            node.is_leaf = True
            return node

        if self.max_depth is not None and depth >= self.max_depth:
            node.is_leaf = True
            return node

        split = self._best_split(X, y)
        if split is None:
            node.is_leaf = True
            return node

        feature_index, threshold, left_mask, right_mask = split
        node.feature_index = int(feature_index)
        node.feature_name = self.feature_names[int(feature_index)] if self.feature_names else None
        node.threshold = float(threshold)
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Optional[tuple[int, float, np.ndarray, np.ndarray]]:
        n_samples, n_features = X.shape
        best_score = np.inf
        best_split = None

        for feature_idx in range(n_features):
            column = X[:, feature_idx]
            values = np.unique(column)
            if values.size <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = column <= threshold
                right_mask = ~left_mask
                if not np.any(left_mask) or not np.any(right_mask):
                    continue

                left_y = y[left_mask]
                right_y = y[right_mask]
                score = (left_y.size / n_samples) * gini(left_y) + (right_y.size / n_samples) * gini(right_y)

                if score < best_score:
                    best_score = score
                    best_split = (feature_idx, float(threshold), left_mask, right_mask)

        return best_split

    def _prune_node(self, node: CartNode, alpha: float) -> tuple[float, int]:
        leaf_error = float(node.sample_count - node.majority_count)
        if node.is_leaf:
            return leaf_error, 1

        left_error, left_leaves = self._prune_node(node.left, alpha)
        right_error, right_leaves = self._prune_node(node.right, alpha)
        subtree_error = left_error + right_error
        subtree_leaves = left_leaves + right_leaves

        # 计算当前内部节点的有效 alpha，决定是否把整棵子树剪成一个叶子。
        gt = (leaf_error - subtree_error) / (subtree_leaves - 1)
        if gt <= alpha:
            node.is_leaf = True
            node.feature_index = None
            node.feature_name = None
            node.threshold = None
            node.left = None
            node.right = None
            return leaf_error, 1

        return subtree_error, subtree_leaves

    def print_tree(self, node: Optional[CartNode] = None, indent: str = "") -> None:
        if self.tree is None:
            raise ValueError("model is not fitted")

        node = node or self.tree
        if node.is_leaf:
            print(f"{indent}Leaf -> {node.prediction}")
            return

        print(f"{indent}{node.feature_name} <= {node.threshold:.6f}")
        self.print_tree(node.left, indent + "  ")
        print(f"{indent}{node.feature_name} > {node.threshold:.6f}")
        self.print_tree(node.right, indent + "  ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CART classification tree with Gini and cost-complexity pruning")
    parser.add_argument("--data", required=True, help="dataset path, supports CSV or JSON")
    parser.add_argument("--alpha", type=float, default=0.0, help="CART pruning alpha")
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--predict-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, feature_names = load_dataset(args.data)

    model = CARTClassifier(
        min_samples_split=args.min_samples_split,
        max_depth=args.max_depth,
    )
    model.fit(X, y, feature_names)

    print("剪枝前:")
    model.print_tree()
    print("训练准确率:", model.score(X, y))

    model.prune(alpha=args.alpha)
    print("剪枝后:")
    model.print_tree()
    print("训练准确率:", model.score(X, y))

    sample = X[args.predict_index]
    print("样本:", sample.tolist())
    print("预测:", model.predict(sample.reshape(1, -1)).tolist())


if __name__ == "__main__":
    main()
