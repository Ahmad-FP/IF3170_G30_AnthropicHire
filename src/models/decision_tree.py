from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models.persistence import PersistableMixin


def _missing(col: np.ndarray) -> np.ndarray:
    if col.dtype.kind == "f":
        return np.isnan(col)
    if col.dtype.kind == "O":
        return np.array([v is None or v != v for v in col], dtype=bool)
    return np.zeros(len(col), dtype=bool)


def _entropy(counts: np.ndarray):
    totals = counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, counts / totals, 0.0)
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=-1)


@dataclass
class _Node:
    value: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    children: dict | None = None
    default: object = None


class DecisionTree(PersistableMixin):
    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y) -> DecisionTree:
        self.classes_, y_enc = np.unique(np.asarray(y), return_inverse=True)
        cols, masks = self._prepare(X, fit=True)
        self.root_ = self._build(cols, masks, y_enc, np.arange(len(y_enc)), 0)
        return self

    def _prepare(self, X, fit: bool):
        if hasattr(X, "iloc"):
            raw = [X.iloc[:, j].to_numpy() for j in range(X.shape[1])]
        else:
            arr = np.asarray(X)
            raw = [arr[:, j] for j in range(arr.shape[1])]
        if fit:
            self._is_numeric_ = [c.dtype.kind in "iuf" for c in raw]
        cols = [c.astype(np.float64, copy=False) if num else c for c, num in zip(raw, self._is_numeric_)]
        return cols, [_missing(c) for c in cols]

    def _build(self, cols, masks, y_enc, indices, depth) -> _Node:
        counts = np.bincount(y_enc[indices], minlength=len(self.classes_)).astype(np.float64)
        node = _Node(value=counts / counts.sum())

        if (
            _entropy(counts) == 0.0
            or len(indices) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        best = self._best_split(cols, masks, y_enc, indices)
        if best is None:
            return node

        node.feature, node.threshold, node.default, groups = best
        node.children = {
            key: self._build(cols, masks, y_enc, indices[sub], depth + 1)
            for key, sub in groups.items()
        }
        return node

    def _best_split(self, cols, masks, y_enc, indices):
        y = y_enc[indices]
        parent_counts = np.bincount(y, minlength=len(self.classes_)).astype(np.float64)
        parent_entropy = float(_entropy(parent_counts))
        best, best_score = None, 1e-12

        for j, is_num in enumerate(self._is_numeric_):
            col, miss = cols[j][indices], masks[j][indices]
            if miss.all():
                continue
            cand = (
                self._numeric_split(col, miss, y, parent_counts, parent_entropy)
                if is_num
                else self._categorical_split(col, miss, y, parent_entropy)
            )
            if cand is not None and cand[0] > best_score:
                best_score = cand[0]
                best = (j, *cand[1:])
        return best

    def _numeric_split(self, col, miss, y, parent_counts, parent_entropy):
        n, n_classes = len(y), len(self.classes_)
        valid = ~miss
        xv, yv = col[valid], y[valid]
        order = np.argsort(xv)
        xs, ys = xv[order], yv[order]
        diff = xs[:-1] != xs[1:]
        if not diff.any():
            return None

        thresholds = 0.5 * (xs[:-1][diff] + xs[1:][diff])
        one_hot = np.zeros((len(ys), n_classes), dtype=np.float64)
        one_hot[np.arange(len(ys)), ys] = 1.0
        left_v = np.cumsum(one_hot, axis=0)[:-1][diff]
        mc = np.bincount(y[miss], minlength=n_classes).astype(np.float64)
        right_v = (parent_counts - mc) - left_v

        dl_arr = left_v.sum(axis=1) >= right_v.sum(axis=1)
        L = left_v + np.where(dl_arr[:, None], mc, 0.0)
        R = right_v + np.where(dl_arr[:, None], 0.0, mc)
        ln, rn = L.sum(axis=1), R.sum(axis=1)
        ok = (ln >= self.min_samples_leaf) & (rn >= self.min_samples_leaf)
        if not ok.any():
            return None

        gains = parent_entropy - (ln[ok] * _entropy(L[ok]) + rn[ok] * _entropy(R[ok])) / n
        idx = int(np.argmax(gains))
        pos = int(np.flatnonzero(ok)[idx])
        thr, dl = float(thresholds[pos]), bool(dl_arr[pos])
        left_mask = np.full(n, dl, dtype=bool)
        left_mask[valid] = xv < thr
        return float(gains[idx]), thr, dl, {True: left_mask, False: ~left_mask}

    def _categorical_split(self, col, miss, y, parent_entropy):
        n, n_classes = len(y), len(self.classes_)
        groups: dict[object, list[int]] = {}
        for pos in np.flatnonzero(~miss):
            groups.setdefault(col[pos], []).append(int(pos))
        if len(groups) < 2:
            return None

        mode_val = max(groups, key=lambda v: len(groups[v]))
        if miss.any():
            groups[mode_val].extend(np.flatnonzero(miss).tolist())
        if any(len(sub) < self.min_samples_leaf for sub in groups.values()):
            return None

        sub_counts = np.array(
            [np.bincount(y[sub], minlength=n_classes) for sub in groups.values()],
            dtype=np.float64,
        )
        weights = np.array([len(sub) for sub in groups.values()], dtype=np.float64) / n
        score = float(parent_entropy - np.dot(weights, _entropy(sub_counts)))
        return score, None, mode_val, groups

    def _predict_row(self, cols, masks, i: int) -> np.ndarray:
        node = self.root_
        while node.feature is not None:
            if masks[node.feature][i]:
                key = node.default
            elif node.threshold is not None:
                key = bool(cols[node.feature][i] < node.threshold)
            else:
                key = cols[node.feature][i]
            child = node.children.get(key)
            if child is None:
                return node.value
            node = child
        return node.value

    def predict_proba(self, X) -> np.ndarray:
        cols, masks = self._prepare(X, fit=False)
        return np.array([self._predict_row(cols, masks, i) for i in range(len(cols[0]))], dtype=np.float64)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
