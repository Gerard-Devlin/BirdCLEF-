from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import roc_auc_score


def macro_auc_skip_empty(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aucs: List[float] = []
    num_classes = y_true.shape[1]

    for class_index in range(num_classes):
        gt = y_true[:, class_index]
        pr = y_pred[:, class_index]

        # Mirrors competition behavior: skip classes without positive labels.
        if gt.sum() == 0:
            continue
        # roc_auc_score also needs both classes in ground truth.
        if np.unique(gt).size < 2:
            continue

        aucs.append(float(roc_auc_score(gt, pr)))

    if not aucs:
        return 0.0
    return float(np.mean(aucs))
