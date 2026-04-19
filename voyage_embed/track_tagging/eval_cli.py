"""Compare reference vs candidate tag JSONL; print sklearn metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def _load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = str(row["track_id"])
            key = "reference" if "reference" in row and row["reference"] is not None else "prediction"
            if "prediction" in row:
                out.setdefault(tid, {})["prediction"] = row["prediction"]
            if "reference" in row and row["reference"] is not None:
                out.setdefault(tid, {})["reference"] = row["reference"]
    return out


def _merge_by_track(ref_path: Path, pred_path: Path) -> list[tuple[str, dict, dict]]:
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
    for k in ("flow", "ritual", "expanded_state"):
        block = themes.get(k) or {}
        out[k] = (bool(block.get("assigned")), float(block.get("confidence") or 0.0))
    return out


def run_eval(pairs: list[tuple[str, dict, dict]]) -> None:
    if not pairs:
        print("No overlapping track_ids with valid dict predictions.")
        return

    for theme in ("flow", "ritual", "expanded_state"):
        y_true = []
        y_score = []
        y_pred = []
        for _, ref, pred in pairs:
            rt = _theme_keys(ref.get("themes"))
            pt = _theme_keys(pred.get("themes"))
            if theme not in rt or theme not in pt:
                continue
            y_true.append(int(rt[theme][0]))
            y_score.append(pt[theme][1])
            y_pred.append(int(pt[theme][0]))
        if not y_true:
            continue
        print(f"\n=== Theme: {theme} ===")
        print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_true, y_pred, labels=[0, 1]))
        print("Accuracy:", accuracy_score(y_true, y_pred))
        try:
            if len(set(y_true)) > 1:
                print("ROC-AUC (confidence vs true assigned):", roc_auc_score(y_true, y_score))
        except ValueError as e:
            print("ROC-AUC skipped:", e)

    # Primary genre exact match
    genres_true = [p[1].get("primary_genre") for p in pairs]
    genres_pred = [p[2].get("primary_genre") for p in pairs]
    print("\n=== primary_genre ===")
    print("Accuracy:", accuracy_score(genres_true, genres_pred))

    # Energy MAE
    e_true = [float(p[1].get("energy_level") or 0) for p in pairs]
    e_pred = [float(p[2].get("energy_level") or 0) for p in pairs]
    print("\n=== energy_level ===")
    print("MAE:", float(np.mean(np.abs(np.array(e_true) - np.array(e_pred)))))


def main() -> None:
    p = argparse.ArgumentParser(description="Eval tag predictions vs reference JSONL.")
    p.add_argument("--reference-jsonl", type=Path, required=True)
    p.add_argument("--candidate-jsonl", type=Path, required=True)
    args = p.parse_args()
    pairs = _merge_by_track(args.reference_jsonl, args.candidate_jsonl)
    print(f"Pairs: {len(pairs)}")
    run_eval(pairs)


if __name__ == "__main__":
    main()
