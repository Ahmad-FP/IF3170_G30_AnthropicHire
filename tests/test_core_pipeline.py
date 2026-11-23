import numpy as np

from scripts.core import evaluate, plot_class_balance, plot_confusion, plot_scores
from src.models import KNN


def _toy_split():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-1, 0.5, (20, 3)), rng.normal(1, 0.5, (20, 3))])
    y = np.array([0] * 20 + [1] * 20)
    return X[:30], y[:30], X[30:], y[30:]


def test_evaluate_and_plots(tmp_path):
    X, y, X_val, y_val = _toy_split()

    table, predictions = evaluate([("knn", KNN(k=3), None)], X, y, X_val, y_val)

    assert list(table["model"]) == ["knn"]
    assert 0.0 <= table.loc[0, "macro_f1"] <= 1.0

    plot_scores(table, tmp_path / "scores.png")
    plot_confusion(y_val, predictions["knn"], "knn", tmp_path / "confusion.png")
    plot_class_balance(y, y_val, tmp_path / "balance.png")

    assert all(path.stat().st_size > 0 for path in tmp_path.iterdir())
