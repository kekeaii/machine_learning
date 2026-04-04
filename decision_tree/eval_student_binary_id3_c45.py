from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from decision_tree_id3_c45 import DecisionTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ID3/C4.5 on processed student CSV")
    parser.add_argument("--data", required=True, help="processed CSV path")
    parser.add_argument("--algo", choices=["id3", "c45"], required=True, help="tree algorithm")
    parser.add_argument("--n-samples", type=int, default=1000, help="use first n samples before shuffle")
    parser.add_argument("--train-size", type=int, default=800, help="number of training samples")
    parser.add_argument("--random-state", type=int, default=42, help="shuffle seed")
    parser.add_argument("--alpha", type=float, default=0.0, help="post-pruning alpha")
    return parser.parse_args()


def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def tree_depth(node) -> int:
    if node.is_leaf:
        return 1
    return 1 + max(tree_depth(child) for child in node.children.values())


def main() -> None:
    args = parse_args()

    raw_df = pd.read_csv(args.data)
    print("处理后数据前5行:")
    print(raw_df.head())
    print()

    df = raw_df.iloc[: args.n_samples].sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    y = df["label"].to_numpy(dtype=object)
    original_feature_df = df.drop(columns=["label"]).copy()
    X_df = original_feature_df.astype(str)

    print("使用的特征列:")
    print(X_df.columns.tolist())
    print()

    X = X_df.to_numpy(dtype=object)
    X_train = X[: args.train_size]
    y_train = y[: args.train_size]
    X_eval = X[args.train_size :]
    y_eval = y[args.train_size :]

    model = DecisionTree(criterion=args.algo)
    model.fit(X_train, y_train, X_df.columns.tolist())
    model.prune(alpha=args.alpha)

    train_pred = model.predict(X_train)
    eval_pred = model.predict(X_eval)

    train_acc = float(np.mean(train_pred == y_train))
    eval_acc = float(np.mean(eval_pred == y_eval))
    eval_metrics = confusion_metrics(y_eval, eval_pred)

    print(f"algo={args.algo}")
    print(f"train_samples={len(y_train)}")
    print(f"eval_samples={len(y_eval)}")
    print(f"train_accuracy={train_acc:.4f}")
    print(f"eval_accuracy={eval_acc:.4f}")
    print(f"tree_depth={tree_depth(model.tree)}")
    print("eval_label_counts=", {int(k): int(v) for k, v in zip(*np.unique(y_eval.astype(int), return_counts=True))})
    print("pred_label_counts=", {int(k): int(v) for k, v in zip(*np.unique(eval_pred.astype(int), return_counts=True))})
    print("confusion_matrix=", eval_metrics)
    print()

    eval_original_df = original_feature_df.iloc[args.train_size :].reset_index(drop=True)
    for i in range(min(2, len(eval_original_df))):
        print(f"sample_{i}:")
        print("x =", eval_original_df.iloc[i].to_dict())
        print("y_true =", int(y_eval[i]))
        print("y_pred =", int(eval_pred[i]))
        print()


if __name__ == "__main__":
    main()
