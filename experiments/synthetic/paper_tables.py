#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper_tables.py  —  Split statistics + baseline evaluation tables for the
                    synthetic BiSBM dataset (ready for paper inclusion).

Outputs two LaTeX-compatible tables per weight condition:

  TABLE 1 – Edge split statistics
    Rows : TRAIN | VAL | TEST P+ | TEST P- | TOTAL
    Cols : O-O | O-U / U-O | U-U | Total

  TABLE 2 – Per-group evaluation on TEST (dot-product baseline over raw features)
    Rows : O-O | O-U | U-U | GLOBAL
    Cols : Nodes U | Nodes V | Pos / Neg | ROC-AUC | AP | F1 | Prec | Recall | Acc

Usage
-----
    python experiments/synthetic/paper_tables.py
    python experiments/synthetic/paper_tables.py --condition informative
    python experiments/synthetic/paper_tables.py --no_eval      # split stats only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
)

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
CONDITIONS = ["informative", "anti_correlated", "orthogonal"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _load(condition: str) -> dict:
    path = DATA_DIR / f"synthetic_{condition}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run:  python data/synthetic/generator.py"
        )
    return dict(np.load(path, allow_pickle=True))


def _type_counts(types: np.ndarray) -> tuple[int, int, int]:
    oo = int((types == "O-O").sum())
    ou = int((types == "O-U").sum())
    uu = int((types == "U-U").sum())
    return oo, ou, uu


def _compute_metrics(labels: np.ndarray, scores: np.ndarray,
                     threshold: float = 0.5) -> dict:
    if np.isnan(scores).any():
        scores = np.nan_to_num(scores, nan=0.5)
    if np.all(scores == scores[0]):
        rng = np.random.default_rng(0)
        scores = scores + rng.normal(0, 1e-8, len(scores))
    try:
        roc = roc_auc_score(labels, scores)
    except ValueError:
        roc = 0.5
    try:
        ap = average_precision_score(labels, scores)
    except ValueError:
        ap = float(np.mean(labels))
    preds = (scores >= threshold).astype(int)
    return {
        "ROC-AUC": roc,
        "AP":      ap,
        "F1":      f1_score(labels, preds, zero_division=0),
        "Prec":    precision_score(labels, preds, zero_division=0),
        "Recall":  recall_score(labels, preds, zero_division=0),
        "Acc":     accuracy_score(labels, preds),
    }


def _dot_scores(u_feat: np.ndarray, v_feat: np.ndarray,
                edge_index: np.ndarray, num_u: int) -> np.ndarray:
    """
    Dot-product score for each edge.
    edge_index: [2, E] — row 0 = local U index, row 1 = local V index (offset by num_u)
    Feature dims may differ (4 vs 3); we use the min dimension.
    """
    u_idx = edge_index[0]
    v_idx = edge_index[1] - num_u
    min_dim = min(u_feat.shape[1], v_feat.shape[1])
    scores = np.zeros(edge_index.shape[1], dtype=np.float64)
    mask = (
        (u_idx >= 0) & (u_idx < len(u_feat)) &
        (v_idx >= 0) & (v_idx < len(v_feat))
    )
    if mask.any():
        scores[mask] = (
            u_feat[u_idx[mask], :min_dim].astype(np.float64)
            * v_feat[v_idx[mask], :min_dim].astype(np.float64)
        ).sum(axis=1)
    return scores


def _generate_typed_negatives(
    pos_edge_index: np.ndarray,   # [2, E_pos]  local indices, V offset by num_u
    pair_types:     np.ndarray,   # string array [E_pos]
    test_u_ids_vocab: np.ndarray, # [N_u]  0=new, >0=old
    test_v_ids_vocab: np.ndarray, # [N_v]  0=new, >0=old
    num_u: int,
    rng: np.random.Generator,
    max_tries: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 1-to-1 typed negative edges (same type distribution as positives).

    Returns
    -------
    neg_edge_index : [2, E_neg]   same shape as pos_edge_index
    neg_types      : [E_neg]      string array matching pos types
    """
    old_u = np.where(test_u_ids_vocab > 0)[0].astype(np.int64)  # local U indices
    new_u = np.where(test_u_ids_vocab == 0)[0].astype(np.int64)
    old_v = np.where(test_v_ids_vocab > 0)[0].astype(np.int64)  # local V indices (0-based in V space)
    new_v = np.where(test_v_ids_vocab == 0)[0].astype(np.int64)

    # Build positive-edge set for fast lookup
    pos_set = set(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))

    neg_us, neg_vs, neg_ts = [], [], []

    for i in range(pos_edge_index.shape[1]):
        t = pair_types[i]
        if t == "O-O":
            pool_u, pool_v_local = old_u, old_v
        elif t == "U-U":
            pool_u, pool_v_local = new_u, new_v
        else:  # O-U  (either side can be new)
            # Use same side assignment as the positive edge
            u_is_old = test_u_ids_vocab[pos_edge_index[0, i]] > 0
            if u_is_old:
                pool_u, pool_v_local = old_u, new_v
            else:
                pool_u, pool_v_local = new_u, old_v

        if len(pool_u) == 0 or len(pool_v_local) == 0:
            # Fallback: any pair not in positive set
            pool_u = np.arange(num_u, dtype=np.int64)
            pool_v_local = np.arange(len(test_v_ids_vocab), dtype=np.int64)

        found = False
        for _ in range(max_tries):
            u = int(rng.choice(pool_u))
            v_local = int(rng.choice(pool_v_local))
            v = v_local + num_u  # restore global local offset
            if (u, v) not in pos_set:
                neg_us.append(u)
                neg_vs.append(v)
                neg_ts.append(t)
                pos_set.add((u, v))  # avoid re-sampling the same negative
                found = True
                break
        if not found:
            # Accept duplicate rather than crash
            neg_us.append(int(rng.choice(pool_u)))
            neg_vs.append(int(rng.choice(pool_v_local)) + num_u)
            neg_ts.append(t)

    neg_edge_index = np.array([neg_us, neg_vs], dtype=np.int64)
    return neg_edge_index, np.array(neg_ts)


# ── table builders ────────────────────────────────────────────────────────────

def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _separator(widths: list[int], char: str = "-") -> str:
    return "+" + "+".join(char * (w + 2) for w in widths) + "+"


def _row(cells: list[str], widths: list[int]) -> str:
    parts = [f" {c:<{widths[i]}} " for i, c in enumerate(cells)]
    return "|" + "|".join(parts) + "|"


def _print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    all_rows = [headers] + rows
    widths = [max(len(str(r[j])) for r in all_rows) for j in range(len(headers))]
    sep = _separator(widths)
    thick = _separator(widths, "=")
    print(f"\n  {title}")
    print(thick)
    print(_row(headers, widths))
    print(thick)
    for i, row in enumerate(rows):
        print(_row(row, widths))
        if i < len(rows) - 1:
            print(sep)
    print(thick)


# ── main analysis per condition ──────────────────────────────────────────────

def analyze_condition(condition: str, do_eval: bool = True) -> None:
    print(f"\n{'#'*70}")
    print(f"  CONDITION: {condition.upper()}")
    print(f"{'#'*70}")

    d = _load(condition)

    # ── TABLE 1: split statistics ─────────────────────────────────────────
    n_train = d["train_edge_index"].shape[1]
    train_oo, train_ou, train_uu = n_train, 0, 0  # training is always O-O

    val_types = d["val_pair_types"]
    val_oo, val_ou, val_uu = _type_counts(val_types)

    test_types = d["test_pair_types"]
    test_oo, test_ou, test_uu = _type_counts(test_types)

    # Negatives: 1:1 per type (same counts as positives)
    neg_oo, neg_ou, neg_uu = test_oo, test_ou, test_uu

    total_oo = train_oo + val_oo + test_oo
    total_ou = train_ou + val_ou + test_ou
    total_uu = train_uu + val_uu + test_uu

    headers_t1 = ["Split", "O-O (old-old)", "O-U / U-O", "U-U", "Total edges"]
    rows_t1 = [
        ["TRAIN",    str(train_oo), str(train_ou), str(train_uu), str(n_train)],
        ["VAL",      str(val_oo),   str(val_ou),   str(val_uu),   str(val_oo + val_ou + val_uu)],
        ["TEST P+",  str(test_oo),  str(test_ou),  str(test_uu),  str(test_oo + test_ou + test_uu)],
        ["TEST P-",  str(neg_oo),   str(neg_ou),   str(neg_uu),   str(neg_oo + neg_ou + neg_uu)],
        ["TOTAL",
         str(total_oo),
         str(total_ou),
         str(total_uu),
         str(total_oo + total_ou + total_uu)],
    ]
    _print_table("TABLE 1 — Edge Split Statistics", headers_t1, rows_t1)

    # Also print proportions
    print(f"\n  Split fractions (of O-O pool = {total_oo}):")
    print(f"    TRAIN : {train_oo}/{total_oo} = {train_oo/total_oo*100:.1f}%")
    print(f"    VAL   : {val_oo}/{total_oo}  = {val_oo/total_oo*100:.1f}%")
    print(f"    TEST  : {test_oo}/{total_oo}  = {test_oo/total_oo*100:.1f}%")
    print(f"  Inductive pool: O-U={val_ou+test_ou} (50/50), U-U={val_uu+test_uu} (50/50)")

    if not do_eval:
        return

    # ── TABLE 2: per-group evaluation ────────────────────────────────────
    test_edge_index = d["test_edge_index"]         # [2, E_test]
    test_u_feat     = d["test_u_features"]         # [N_u, 4]
    test_v_feat     = d["test_v_features"]         # [N_v, 3]
    test_u_ids_v    = d["test_u_ids"]              # vocab ids [N_u]
    test_v_ids_v    = d["test_v_ids"]              # vocab ids [N_v]
    num_u           = len(test_u_feat)
    num_v           = len(test_v_feat)

    rng = np.random.default_rng(0)
    neg_edge_index, neg_types = _generate_typed_negatives(
        test_edge_index, test_types,
        test_u_ids_v, test_v_ids_v,
        num_u, rng,
    )

    # Score positive and negative edges
    pos_scores = _dot_scores(test_u_feat, test_v_feat, test_edge_index, num_u)
    neg_scores = _dot_scores(test_u_feat, test_v_feat, neg_edge_index, num_u)

    def _eval_subset(mask_pos, mask_neg):
        if mask_pos.sum() == 0:
            return None
        s = np.concatenate([pos_scores[mask_pos], neg_scores[mask_neg]])
        l = np.concatenate([
            np.ones(mask_pos.sum(), dtype=np.int32),
            np.zeros(mask_neg.sum(), dtype=np.int32),
        ])
        return _compute_metrics(l, s)

    groups = {
        "O-O":    (test_types == "O-O", neg_types == "O-O"),
        "O-U":    (test_types == "O-U", neg_types == "O-U"),
        "U-U":    (test_types == "U-U", neg_types == "U-U"),
        "GLOBAL": (np.ones(len(test_types), dtype=bool),
                   np.ones(len(neg_types),  dtype=bool)),
    }

    # Count unique nodes per group for info
    def _count_nodes(mask, edge_index, nu):
        idx = edge_index[:, mask]
        n_u = len(np.unique(idx[0]))
        n_v = len(np.unique(idx[1] - nu))
        return n_u, n_v

    met_key_order = ["ROC-AUC", "AP", "F1", "Prec", "Recall", "Acc"]
    headers_t2 = ["Group", "Nodes U", "Nodes V", "Pos / Neg"] + met_key_order
    rows_t2 = []

    for grp_name, (pmask, nmask) in groups.items():
        metrics = _eval_subset(pmask, nmask)
        n_pos = int(pmask.sum())
        n_neg = int(nmask.sum())
        if test_edge_index.shape[1] > 0 and pmask.any():
            nu, nv = _count_nodes(pmask, test_edge_index, num_u)
        else:
            nu, nv = 0, 0
        row = [
            grp_name,
            str(nu),
            str(nv),
            f"{n_pos} / {n_neg}",
        ]
        if metrics:
            row += [_fmt_pct(metrics[k]) for k in met_key_order]
        else:
            row += ["N/A"] * len(met_key_order)
        rows_t2.append(row)

    _print_table(
        "TABLE 2 — Per-Group Evaluation on TEST (dot-product baseline)",
        headers_t2, rows_t2,
    )

    # ── LaTeX snippet ─────────────────────────────────────────────────────
    print(f"\n  [LaTeX TABLE 1 — {condition}]")
    print(r"  \begin{tabular}{lrrrr}")
    print(r"  \toprule")
    print(r"  Split & O--O & O--U/U--O & U--U & Total \\")
    print(r"  \midrule")
    for row in rows_t1[:-1]:      # data rows (not TOTAL)
        print(f"  {row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\")
    print(r"  \midrule")
    tot = rows_t1[-1]
    print(f"  \\textbf{{Total}} & {tot[1]} & {tot[2]} & {tot[3]} & {tot[4]} \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")

    print(f"\n  [LaTeX TABLE 2 — {condition}]")
    print(r"  \begin{tabular}{lrrr" + "r" * len(met_key_order) + "}")
    print(r"  \toprule")
    latex_header = (
        "  Group & Nodes-U & Nodes-V & Pos/Neg & "
        + " & ".join(met_key_order)
        + r" \\"
    )
    print(latex_header)
    print(r"  \midrule")
    for row in rows_t2[:-1]:  # data rows (not GLOBAL)
        print("  " + " & ".join(row) + r" \\")
    print(r"  \midrule")
    glb = rows_t2[-1]
    print("  \\textbf{Global} & " + " & ".join(glb[1:]) + r" \\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-ready tables for the synthetic BiSBM dataset."
    )
    parser.add_argument(
        "--condition", nargs="*", choices=CONDITIONS, default=None,
        help="Weight condition(s) to analyse. Default: all three.",
    )
    parser.add_argument(
        "--no_eval", action="store_true",
        help="Print split statistics only (skip evaluation).",
    )
    args = parser.parse_args()

    conds = args.condition or CONDITIONS
    for c in conds:
        analyze_condition(c, do_eval=not args.no_eval)

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
