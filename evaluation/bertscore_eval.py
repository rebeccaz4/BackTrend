"""BERTScore-based evaluation: Setting 1 (set-level) and Setting 3 (signal-level).

Setting 1 — Set-level BERTScore:
    Concatenate all signals in each set into one string, then compute BERTScore
    on the two concatenated strings. Returns a single (P, R, F1).

Setting 3 — Signal-level BERTScore:
    Compute BERTScore F1 for every (pred_i, gt_j) pair to build an n×m matrix S.
    Set-Precision = (1/n) * sum_i  max_j S[i,j]
    Set-Recall    = (1/m) * sum_j  max_i S[i,j]
    F1            = harmonic mean of Precision and Recall
"""
from __future__ import annotations

from bert_score import score as _bert_score

MODEL_TYPE = "roberta-large"
SEP = " [SEP] "


def _bs(cands: list[str], refs: list[str]) -> tuple[list[float], list[float], list[float]]:
    """Call bert_score and return (P, R, F1) as plain Python lists."""
    P, R, F1 = _bert_score(
        cands,
        refs,
        lang="en",
        model_type=MODEL_TYPE,
        rescale_with_baseline=False,
        verbose=False,
    )
    return P.tolist(), R.tolist(), F1.tolist()


# ---------------------------------------------------------------------------
# Setting 1 — Set-level BERTScore
# ---------------------------------------------------------------------------

def eval_set_bertscore(gt: list[str], pred: list[str]) -> dict:
    """Setting 1: concatenate each set into one string, compute BERTScore."""
    gt_text = SEP.join(gt)
    pred_text = SEP.join(pred)
    P, R, F1 = _bs([pred_text], [gt_text])
    p, r, f = round(P[0], 4), round(R[0], 4), round(F1[0], 4)
    return {"setting": "set_bertscore", "precision": p, "recall": r, "f1": f}


# ---------------------------------------------------------------------------
# Setting 3 — Signal-level BERTScore (greedy max over n×m matrix)
# ---------------------------------------------------------------------------

def eval_signal_bertscore(gt: list[str], pred: list[str]) -> dict:
    """Setting 3: pairwise BERTScore F1 matrix, aggregated via greedy max."""
    n, m = len(pred), len(gt)
    if n == 0 or m == 0:
        return {
            "setting": "signal_bertscore",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_pred": n,
            "n_gt": m,
        }

    # Build all n*m (pred, gt) pairs in row-major order: pair (i*m+j) = (pred[i], gt[j])
    cands = [p for p in pred for _ in gt]
    refs  = [g for _ in pred for g in gt]

    _, _, F1_flat = _bs(cands, refs)
    # Reshape to (n, m) using plain Python lists
    S = [F1_flat[i * m : (i + 1) * m] for i in range(n)]  # S[i][j] = F1(pred[i], gt[j])

    # Precision: for each prediction, max similarity over all GT signals
    precision_scores = [max(S[i]) for i in range(n)]
    # Recall: for each GT signal, max similarity over all predictions
    recall_scores = [max(S[i][j] for i in range(n)) for j in range(m)]
    precision_raw = sum(precision_scores) / n
    recall_raw = sum(recall_scores) / m
    denom = precision_raw + recall_raw
    f1_raw = 2 * precision_raw * recall_raw / denom if denom > 0 else 0.0
    precision = round(precision_raw, 4)
    recall = round(recall_raw, 4)
    f1 = round(f1_raw, 4)

    return {
        "setting": "signal_bertscore",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": n,
        "n_gt": m,
    }
