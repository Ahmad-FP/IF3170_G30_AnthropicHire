"""End-to-end pipeline: load -> prepare -> from-scratch models -> figures.

Run from the project root:

    .venv/bin/python scripts/core.py

It reads raw CSVs when `data/raw/` exists, otherwise falls back to the
prepared split in `output/prepared/`. Results go to `output/core_*` and
figures to `output/figures/`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessing import ID_COL, TARGET, DataPreprocessor  # noqa: E402
from src.models import (  # noqa: E402
    DecisionTree,
    KNN,
    LogisticRegression,
    NaiveBayes,
    SVM,
)

RAW = ROOT / "data" / "raw"
PREP = ROOT / "output" / "prepared"
OUT = ROOT / "output"
FIG = OUT / "figures"

SEED = 42
KNN_TRAIN_CAP = 5000  # exact nearest-neighbour search is O(n*m)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Raw CSV path (full pipeline) or prepared split fallback."""
    if (RAW / "train.csv").exists():
        train_df = pd.read_csv(RAW / "train.csv", dtype={ID_COL: str})
        prep = DataPreprocessor(seed=SEED, select_k=38)
        train_set, val_set = prep.split(train_df)
        X_train, y_train = prep.fit_transform(train_set)
        return X_train, y_train, prep.transform(val_set), val_set[TARGET].to_numpy()

    train, val = np.load(PREP / "train.npz"), np.load(PREP / "val.npz")
    return train["X"], train["y"], val["X"], val["y"]


def build_models() -> list[tuple[str, object, int | None]]:
    """(name, model, train cap). Caps only limit memory-heavy models."""
    return [
        ("knn", KNN(k=9, p=2, weights="distance"), KNN_TRAIN_CAP),
        ("decision_tree", DecisionTree(criterion="entropy", max_depth=12, min_samples_leaf=5), None),
        (
            "logistic_regression",
            LogisticRegression(
                learning_rate=0.5,
                epochs=1000,
                regularization="l2",
                lambda_=1e-3,
                class_weight="balanced",
            ),
            None,
        ),
        ("svm_linear", SVM(kernel="linear", C=1.0, learning_rate=0.002, epochs=50), None),
        (
            "svm_polynomial",
            SVM(
                kernel="poly",
                degree=3,
                coef0=1.0,
                gamma=0.1,
                C=1.0,
                learning_rate=0.05,
                epochs=30,
                max_samples=2000,
            ),
            None,
        ),
        (
            "svm_gaussian",
            SVM(
                kernel="gaussian",
                gamma=0.15,
                C=1.0,
                learning_rate=0.05,
                epochs=30,
                max_samples=2000,
            ),
            None,
        ),
        ("naive_bayes", NaiveBayes(var_smoothing=1e-6), None),
    ]


def _capped_index(y: np.ndarray, cap: int | None, seed: int) -> np.ndarray:
    if cap is None or len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        take = max(1, round(cap * len(idx) / len(y)))
        chosen.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    return np.sort(np.concatenate(chosen))


def evaluate(models, X, y, X_val, y_val, seed: int = SEED):
    """Fit every (name, model, cap) on train and score it on the same val set."""
    rows, predictions = [], {}
    for name, model, cap in models:
        idx = _capped_index(y, cap, seed)
        t0 = time.perf_counter()
        model.fit(X[idx], y[idx])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = model.predict(X_val)
        predict_s = time.perf_counter() - t0
        rows.append(
            {
                "model": name,
                "train_rows": len(idx),
                "fit_s": fit_s,
                "predict_s": predict_s,
                "macro_f1": f1_score(y_val, pred, average="macro"),
            }
        )
        predictions[name] = pred
        print(
            f"{name:<20} train={len(idx):>6}  fit={fit_s:6.2f}s  "
            f"pred={predict_s:6.2f}s  macro-F1={rows[-1]['macro_f1']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False), predictions


def plot_scores(table: pd.DataFrame, path: Path) -> None:
    ordered = table.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ordered) + 1.5))
    bars = ax.barh(ordered["model"], ordered["macro_f1"], color="#4C72B0")
    bars[-1].set_color("#DD8452")
    for bar, score in zip(bars, ordered["macro_f1"]):
        ax.text(score + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va="center")
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro F1 (validation)")
    ax.set_title("From-scratch models")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, y_pred, name: str, path: Path) -> None:
    labels = np.unique(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{name} — confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_balance(y_train, y_val, path: Path) -> None:
    classes = np.unique(y_train)
    train_share = [np.mean(y_train == c) for c in classes]
    val_share = [np.mean(y_val == c) for c in classes]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(x - 0.2, train_share, 0.4, label="train")
    ax.bar(x + 0.2, val_share, 0.4, label="validation")
    ax.set_xticks(x, [str(c) for c in classes])
    ax.set_xlabel(TARGET)
    ax.set_ylabel("share of rows")
    ax.set_title("Class balance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    X_train, y_train, X_val, y_val = load_data()
    print(f"train {X_train.shape} | val {X_val.shape} | features {X_train.shape[1]}")

    table, predictions = evaluate(build_models(), X_train, y_train, X_val, y_val)
    csv_path = OUT / "core_model_comparison.csv"
    table.to_csv(csv_path, index=False)

    best = str(table.iloc[0]["model"])
    plot_scores(table, FIG / "model_comparison.png")
    for name in table["model"]:
        plot_confusion(y_val, predictions[name], name, FIG / f"confusion_{name}.png")
    plot_class_balance(y_train, y_val, FIG / "class_balance.png")

    print(f"\nbest: {best} (macro F1 {table.iloc[0]['macro_f1']:.4f})")
    print(f"wrote {csv_path}")
    print(f"figures: {FIG}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
