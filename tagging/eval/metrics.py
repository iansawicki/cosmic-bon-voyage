"""Offline metrics for reference vs candidate tag JSONL (JSON-serializable dicts)."""

from __future__ import annotations

import csv
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

EPS = 1e-12

THEME_KEYS = ("flow", "ritual", "expanded_state")
LIST_FIELDS = ("style_tags", "mood_keywords", "search_keywords")


def merge_by_track(ref_path: Path, pred_path: Path) -> list[tuple[str, dict, dict]]:
    """Load reference and candidate JSONL; return (track_id, ref_tags, pred_tags) for overlapping ids."""
    ref: dict[str, dict] = {}
    with ref_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = str(row["track_id"])
            ref[tid] = row.get("reference") or row.get("prediction") or row.get("tags")

    pairs: list[tuple[str, dict, dict]] = []
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = str(row["track_id"])
            pred = row.get("prediction") or row.get("tags")
            if tid in ref and isinstance(ref[tid], dict) and isinstance(pred, dict):
                pairs.append((tid, ref[tid], pred))
    return pairs


def _theme_keys(themes: dict | None) -> dict[str, tuple[bool, float]]:
    if not themes:
        return {}
    out = {}
    for k in THEME_KEYS:
        block = themes.get(k) or {}
        out[k] = (bool(block.get("assigned")), float(block.get("confidence") or 0.0))
    return out


def _norm_str(s: Any) -> str:
    return str(s or "").strip().lower()


def _norm_tag_list(xs: Any) -> list[str]:
    if not isinstance(xs, list):
        return []
    out: list[str] = []
    for x in xs:
        s = str(x).strip().lower()
        if s:
            out.append(s)
    return out


def _jaccard_sets(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sanitize(obj: Any) -> Any:
    """Make structures JSON-safe (no numpy scalars)."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def compute_all_metrics(
    pairs: list[tuple[str, dict, dict]],
    *,
    genre_top_k: int = 30,
    genre_max_labels: int = 50,
) -> dict[str, Any]:
    """
    Compute all metrics for aligned (track_id, reference, prediction) rows.

    List fields are compared with strings normalized to strip + lower.
    """
    out: dict[str, Any] = {"pair_count": len(pairs)}
    if not pairs:
        return _sanitize(out)

    # sklearn emits UserWarning when only one class appears even with explicit labels=
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*single label was found.*", category=UserWarning)

        # --- Themes ---
        themes_out: dict[str, Any] = {}
        for theme in THEME_KEYS:
            y_true: list[int] = []
            y_score: list[float] = []
            y_pred: list[int] = []
            for _, ref, pred in pairs:
                rt = _theme_keys(ref.get("themes"))
                pt = _theme_keys(pred.get("themes"))
                if theme not in rt or theme not in pt:
                    continue
                y_true.append(int(rt[theme][0]))
                y_score.append(pt[theme][1])
                y_pred.append(int(pt[theme][0]))
            if not y_true:
                themes_out[theme] = {"skipped": "no overlapping rows for this theme"}
                continue
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_list = [[int(x) for x in row] for row in cm.tolist()]
            tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
            acc = float(accuracy_score(y_true, y_pred))
            bal = float(balanced_accuracy_score(y_true, y_pred))
            roc: float | None = None
            roc_skip: str | None = None
            try:
                if len(set(y_true)) > 1:
                    roc = float(roc_auc_score(y_true, y_score))
            except ValueError as e:
                roc_skip = str(e)
            themes_out[theme] = {
                "confusion_matrix": cm_list,
                "labels": [0, 1],
                "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
                "accuracy": acc,
                "balanced_accuracy": bal,
                "roc_auc": roc,
                "roc_auc_skipped": roc_skip,
                "n": len(y_true),
            }
        out["themes"] = themes_out

        # --- primary_genre ---
        genres_true = [_norm_str(p[1].get("primary_genre")) for p in pairs]
        genres_pred = [_norm_str(p[2].get("primary_genre")) for p in pairs]
        ref_ctr = Counter(genres_true)
        labels_for_report = [g for g, _ in ref_ctr.most_common(genre_max_labels)]
        if not labels_for_report:
            labels_for_report = sorted(set(genres_true))

        genre_block: dict[str, Any] = {
            "accuracy": float(accuracy_score(genres_true, genres_pred)),
            "genre_top_k": genre_top_k,
            "genre_max_labels": genre_max_labels,
        }
        cr = classification_report(
            genres_true,
            genres_pred,
            labels=labels_for_report,
            zero_division=0,
            output_dict=True,
        )
        genre_block["classification_report"] = _sanitize(cr)

        # Top-K bucketing for confusion matrix
        top_k_labels = [g for g, _ in ref_ctr.most_common(genre_top_k)]
        top_set = set(top_k_labels)

        def bucket(g: str) -> str:
            return g if g in top_set else "__other__"

        y_b_true = [bucket(g) for g in genres_true]
        y_b_pred = [bucket(g) for g in genres_pred]
        all_labels = sorted(set(y_b_true) | set(y_b_pred))
        if len(all_labels) > 1 or (len(all_labels) == 1 and all_labels[0] != "__other__"):
            cm_g = confusion_matrix(y_b_true, y_b_pred, labels=all_labels)
            genre_block["confusion_top_k"] = {
                "matrix": [[int(x) for x in row] for row in cm_g.tolist()],
                "labels": all_labels,
                "note": "Rows and columns are true vs predicted; labels are top-K reference genres plus __other__.",
            }
        else:
            genre_block["confusion_top_k"] = {"matrix": [], "labels": [], "note": "insufficient label diversity"}

        out["primary_genre"] = genre_block

    # --- List fields + secondary_genres ---
    lists: dict[str, Any] = {}

    def field_list_metrics(field: str) -> dict[str, Any]:
        jaccards: list[float] = []
        sum_inter = 0
        sum_pred = 0
        sum_ref = 0
        for _, ref, pred in pairs:
            r = set(_norm_tag_list(ref.get(field)))
            p = set(_norm_tag_list(pred.get(field)))
            jaccards.append(_jaccard_sets(r, p))
            sum_inter += len(r & p)
            sum_pred += len(p)
            sum_ref += len(r)
        prec = float(sum_inter / max(sum_pred, EPS))
        rec = float(sum_inter / max(sum_ref, EPS))
        f1 = float(2 * prec * rec / max(prec + rec, EPS))
        return {
            "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
            "precision_micro": prec,
            "recall_micro": rec,
            "f1_micro": f1,
        }

    for field in LIST_FIELDS:
        lists[field] = field_list_metrics(field)

    # secondary_genres: max 2 items, set comparison (same micro + Jaccard)
    jaccards_sec: list[float] = []
    sum_inter = 0
    sum_pred = 0
    sum_ref = 0
    for _, ref, pred in pairs:
        r = set(_norm_tag_list(ref.get("secondary_genres")))
        p = set(_norm_tag_list(pred.get("secondary_genres")))
        jaccards_sec.append(_jaccard_sets(r, p))
        sum_inter += len(r & p)
        sum_pred += len(p)
        sum_ref += len(r)
    prec_s = float(sum_inter / max(sum_pred, EPS))
    rec_s = float(sum_inter / max(sum_ref, EPS))
    f1_s = float(2 * prec_s * rec_s / max(prec_s + rec_s, EPS))
    lists["secondary_genres"] = {
        "mean_jaccard": float(np.mean(jaccards_sec)) if jaccards_sec else 0.0,
        "precision_micro": prec_s,
        "recall_micro": rec_s,
        "f1_micro": f1_s,
    }
    out["lists"] = lists

    # --- energy_level ---
    e_true = [float(p[1].get("energy_level") or 0) for p in pairs]
    e_pred = [float(p[2].get("energy_level") or 0) for p in pairs]
    arr_t = np.asarray(e_true, dtype=float)
    arr_p = np.asarray(e_pred, dtype=float)
    mae = float(np.mean(np.abs(arr_t - arr_p)))
    pearson: float | None = None
    if len(arr_t) >= 2 and np.std(arr_t) > 0 and np.std(arr_p) > 0:
        pearson = float(np.corrcoef(arr_t, arr_p)[0, 1])
    out["energy_level"] = {"mae": mae, "pearson_r": pearson}

    return _sanitize(out)


def format_metrics_summary(m: dict[str, Any]) -> str:
    """Human-readable block matching prior CLI style."""
    lines: list[str] = []
    n = m.get("pair_count", 0)
    lines.append(f"Pairs: {n}")
    if not n:
        return "\n".join(lines)

    th = m.get("themes") or {}
    for theme in THEME_KEYS:
        block = th.get(theme) or {}
        if block.get("skipped"):
            continue
        lines.append(f"\n=== Theme: {theme} ===")
        lines.append(f"Confusion matrix (rows=true, cols=pred): {block.get('confusion_matrix')}")
        lines.append(f"Accuracy: {block.get('accuracy')}")
        lines.append(f"Balanced accuracy: {block.get('balanced_accuracy')}")
        if block.get("roc_auc") is not None:
            lines.append(f"ROC-AUC (confidence vs true assigned): {block['roc_auc']}")
        elif block.get("roc_auc_skipped"):
            lines.append(f"ROC-AUC skipped: {block['roc_auc_skipped']}")

    pg = m.get("primary_genre") or {}
    lines.append("\n=== primary_genre ===")
    lines.append(f"Accuracy: {pg.get('accuracy')}")

    lst = m.get("lists") or {}
    for name in (*LIST_FIELDS, "secondary_genres"):
        b = lst.get(name) or {}
        lines.append(f"\n=== {name} ===")
        lines.append(f"Mean Jaccard: {b.get('mean_jaccard')}")
        lines.append(f"Micro precision / recall / F1: {b.get('precision_micro')} / {b.get('recall_micro')} / {b.get('f1_micro')}")

    en = m.get("energy_level") or {}
    lines.append("\n=== energy_level ===")
    lines.append(f"MAE: {en.get('mae')}")
    lines.append(f"Pearson r: {en.get('pearson_r')}")

    return "\n".join(lines)


def write_genre_confusion_csv(path: Path, genre_block: dict[str, Any]) -> None:
    """Write confusion matrix for top-K bucketed genres as CSV."""
    ctk = genre_block.get("confusion_top_k") or {}
    labels = ctk.get("labels") or []
    matrix = ctk.get("matrix") or []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *labels])
        for i, row in enumerate(matrix):
            lab = labels[i] if i < len(labels) else f"row_{i}"
            w.writerow([lab, *[str(x) for x in row]])
