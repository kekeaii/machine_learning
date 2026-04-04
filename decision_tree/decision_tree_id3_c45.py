from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def entropy(labels: np.ndarray) -> float:
    """经验熵 H(D)。"""
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


def majority_label(labels: np.ndarray) -> Any:
    """返回多数类，叶子节点预测时使用。"""
    values, counts = np.unique(labels, return_counts=True)
    value = values[np.argmax(counts)]
    return value.item() if hasattr(value, "item") else value


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    读数据:
    - CSV: pandas 只负责读取，最后一列默认是标签 y
    - JSON: 支持 {"X": ..., "y": ...} 或 [{"x": ..., "y": ...}, ...]
    返回值统一转成 numpy 数组。
    """
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least one feature column and one label column")
        feature_names = df.columns[:-1].tolist()
        X = df.iloc[:, :-1].to_numpy(dtype=object)
        y = df.iloc[:, -1].to_numpy(dtype=object)
        return X, y, feature_names

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "X" in data and "y" in data:
        X = np.asarray(data["X"], dtype=object)
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
        X_arr = np.asarray(X, dtype=object)
        y_arr = np.asarray(y, dtype=object)
        feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
        return X_arr, y_arr, feature_names

    raise ValueError("unsupported dataset format")


@dataclass
class TreeNode:
    is_leaf: bool
    prediction: Any
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    children: Dict[Any, "TreeNode"] = field(default_factory=dict)


class DecisionTree:
    """
    手写分类决策树，只处理离散特征。
    - ID3: 信息增益
    - C4.5: 增益率
    剪枝: 基于经验熵的代价复杂度后剪枝
    """

    def __init__(self, criterion: str = "id3") -> None:
        if criterion not in {"id3", "c45"}:
            raise ValueError("criterion must be 'id3' or 'c45'")
        self.criterion = criterion
        self.tree: Optional[TreeNode] = None
        self.feature_names: Optional[list[str]] = None
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> None:
        if X.size == 0 or y.size == 0:
            raise ValueError("X and y must not be empty")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same length")

        self._train_X = np.asarray(X, dtype=object)
        self._train_y = np.asarray(y, dtype=object)
        n_features = self._train_X.shape[1]
        self.feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]
        feature_indices = np.arange(n_features, dtype=int)
        self.tree = self._build_tree(self._train_X, self._train_y, feature_indices)

    def predict_one(self, sample: np.ndarray) -> Any:
        if self.tree is None:
            raise ValueError("model is not fitted")

        node = self.tree
        while not node.is_leaf:
            value = sample[node.feature_index]
            if value not in node.children:
                return node.prediction
            node = node.children[value]
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=object)
        return np.asarray([self.predict_one(row) for row in X_arr], dtype=object)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        X_arr = np.asarray(X, dtype=object)
        y_arr = np.asarray(y, dtype=object)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same length")
        if X_arr.shape[0] == 0:
            return 0.0
        predictions = self.predict(X_arr)
        return float(np.mean(predictions == y_arr))

    def prune(self, alpha: float = 0.0) -> None:
        """
        自底向上比较两种代价:
        - 保留子树: sum(|N_i| * H(N_i)) + alpha * 叶子数
        - 剪成叶子: |N_t| * H(N_t) + alpha
        若剪成叶子的代价更小，就回缩这个节点。
        """
        if self.tree is None:
            raise ValueError("model is not fitted")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self._train_X is None or self._train_y is None:
            raise ValueError("training data is unavailable for pruning")

        self._prune_node(self.tree, self._train_X, self._train_y, alpha)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> TreeNode:
        prediction = majority_label(y)

        if np.unique(y).size == 1:
            return TreeNode(is_leaf=True, prediction=prediction)

        if feature_indices.size == 0:
            return TreeNode(is_leaf=True, prediction=prediction)

        best_feature = self._choose_best_feature(X, y, feature_indices)
        if best_feature is None:
            return TreeNode(is_leaf=True, prediction=prediction)

        node = TreeNode(
            is_leaf=False,
            prediction=prediction,
            feature_index=int(best_feature),
            feature_name=self.feature_names[int(best_feature)] if self.feature_names else None,
        )

        remaining_features = feature_indices[feature_indices != best_feature]
        values = np.unique(X[:, best_feature])

        for value in values:
            mask = X[:, best_feature] == value
            if np.any(mask):
                child_key = value.item() if hasattr(value, "item") else value
                node.children[child_key] = self._build_tree(
                    X[mask],
                    y[mask],
                    remaining_features,
                )

        if not node.children:
            return TreeNode(is_leaf=True, prediction=prediction)
        return node

    def _choose_best_feature(self, X: np.ndarray, y: np.ndarray, feature_indices: np.ndarray) -> Optional[int]:
        base_entropy = entropy(y)
        best_feature = None
        best_score = -np.inf

        for feature_idx in feature_indices:
            column = X[:, feature_idx]
            values, counts = np.unique(column, return_counts=True)
            weights = counts / counts.sum()

            conditional_entropy = 0.0
            for value, weight in zip(values, weights):
                subset_labels = y[column == value]
                conditional_entropy += weight * entropy(subset_labels)

            info_gain = base_entropy - conditional_entropy
            if self.criterion == "id3":
                score = info_gain
            else:
                split_info = float(-np.sum(weights * np.log2(weights)))
                score = info_gain / split_info if split_info > 0 else 0.0

            if score > best_score:
                best_score = score
                best_feature = int(feature_idx)

        return best_feature

    def _prune_node(self, node: TreeNode, X_sub: np.ndarray, y_sub: np.ndarray, alpha: float) -> tuple[float, int]:
        if y_sub.size == 0:
            return 0.0, 0

        if node.is_leaf:
            return float(y_sub.size * entropy(y_sub) + alpha), 1

        subtree_cost = 0.0
        subtree_leaves = 0

        # 按当前划分特征把训练子集分给各个孩子，再递归计算子树代价。
        for value, child in list(node.children.items()):
            mask = X_sub[:, node.feature_index] == value
            child_cost, child_leaves = self._prune_node(child, X_sub[mask], y_sub[mask], alpha)
            subtree_cost += child_cost
            subtree_leaves += child_leaves

        leaf_cost = float(y_sub.size * entropy(y_sub) + alpha)
        if leaf_cost <= subtree_cost:
            node.is_leaf = True
            node.feature_index = None
            node.feature_name = None
            node.children = {}
            return leaf_cost, 1

        return subtree_cost, subtree_leaves

    def print_tree(self, node: Optional[TreeNode] = None, indent: str = "") -> None:
        if self.tree is None:
            raise ValueError("model is not fitted")

        node = node or self.tree
        if node.is_leaf:
            print(f"{indent}Leaf -> {node.prediction}")
            return

        print(f"{indent}{node.feature_name}")
        for value, child in node.children.items():
            print(f"{indent}  [{value}]")
            self.print_tree(child, indent + "    ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ID3/C4.5 classification tree for discrete features")
    parser.add_argument("--data", required=True, help="dataset path, supports CSV or JSON")
    parser.add_argument("--criterion", choices=["id3", "c45"], default="id3")
    parser.add_argument("--alpha", type=float, default=0.0, help="post-pruning penalty")
    parser.add_argument("--predict-index", type=int, default=0, help="which sample in dataset to predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y, feature_names = load_dataset(args.data)

    model = DecisionTree(criterion=args.criterion)
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
