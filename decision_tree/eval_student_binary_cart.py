from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cart_classification import CARTClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CART on processed student CSV")
    parser.add_argument("--data", required=True, help="processed CSV path")
    parser.add_argument("--n-samples", type=int, default=1000, help="use first n samples before shuffle")
    parser.add_argument("--train-size", type=int, default=800, help="number of training samples")
    parser.add_argument("--random-state", type=int, default=42, help="shuffle seed")
    parser.add_argument("--alpha", type=float, default=0.0, help="CART pruning alpha")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_df = pd.read_csv(args.data)
    print("处理后数据前5行:")
    print(raw_df.head())
    print()

    df = raw_df.iloc[: args.n_samples].sample(frac=1, random_state=args.random_state).reset_index(drop=True)
    y = df["label"].to_numpy(dtype=int)
    original_feature_df = df.drop(columns=["label"]).copy()
    X_df = original_feature_df.copy()

    print("使用的特征列:")
    print(X_df.columns.tolist())
    print()

    for col in X_df.columns:
        if X_df[col].dtype == object:
            X_df[col] = pd.factorize(X_df[col])[0]

    X = X_df.to_numpy(dtype=float)
    X_train = X[: args.train_size]
    y_train = y[: args.train_size]
    X_eval = X[args.train_size :]
    y_eval = y[args.train_size :]

    model = CARTClassifier(min_samples_split=2)
    model.fit(X_train, y_train, feature_names=X_df.columns.tolist())
    model.prune(alpha=args.alpha)

    pred = model.predict(X_eval)
    acc = np.mean(pred == y_eval)

    print(f"train_samples={len(y_train)}")
    print(f"eval_samples={len(y_eval)}")
    print(f"eval_accuracy={acc:.4f}")
    print("eval_label_counts=", {int(k): int(v) for k, v in zip(*np.unique(y_eval, return_counts=True))})
    print("pred_label_counts=", {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))})
    print()

    eval_original_df = original_feature_df.iloc[args.train_size :].reset_index(drop=True)
    for i in range(min(2, len(eval_original_df))):
        print(f"sample_{i}:")
        print("x =", eval_original_df.iloc[i].to_dict())
        print("y_true =", int(y_eval[i]))
        print("y_pred =", int(pred[i]))
        print()


if __name__ == "__main__":
    main()
