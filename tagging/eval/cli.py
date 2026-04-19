"""CLI: compare reference vs candidate tag JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tagging.eval.metrics import (
    compute_all_metrics,
    format_metrics_summary,
    merge_by_track,
    write_genre_confusion_csv,
)


def _plot_matrix(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.35), max(3, len(labels) * 0.35)))
    im = ax.imshow(arr, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    tick_marks = range(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    thresh = arr.max() / 2.0 if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                format(int(arr[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if arr[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Eval tag predictions vs reference JSONL. "
            "Each line must be JSON with track_id; reference lines use 'reference' (or 'tags'); "
            "candidate lines use 'prediction' (or 'tags'). List strings are compared as strip+lower."
        ),
    )
    p.add_argument("--reference-jsonl", type=Path, required=True)
    p.add_argument("--candidate-jsonl", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None, help="Write metrics.json, summary.txt, optional plots/CSV")
    p.add_argument("--genre-top-k", type=int, default=30, help="Top-K reference genres for confusion bucketing (__other__ for the rest)")
    p.add_argument("--genre-max-labels", type=int, default=50, metavar="N", help="Max classes in classification_report")
    p.add_argument(
        "--genre-confusion-csv",
        type=Path,
        default=None,
        help="Write top-K genre confusion matrix CSV here (default: <out-dir>/genre_confusion.csv when --out-dir set)",
    )
    p.add_argument("--plot", action="store_true", help="Save matplotlib heatmaps (requires --out-dir)")
    p.add_argument("--no-plot", action="store_true", help="Disable plots even if --out-dir is set")
    p.add_argument("--plot-max-labels", type=int, default=25, help="Skip genre heatmap if unique bucketed labels exceed this")
    p.add_argument("--quiet", action="store_true", help="With --out-dir, suppress stdout (files only)")
    args = p.parse_args()

    do_plot = bool(args.out_dir and args.plot and not args.no_plot)

    pairs = merge_by_track(args.reference_jsonl, args.candidate_jsonl)
    metrics = compute_all_metrics(
        pairs,
        genre_top_k=args.genre_top_k,
        genre_max_labels=args.genre_max_labels,
    )
    summary_text = format_metrics_summary(metrics)

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (args.out_dir / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")

        csv_path = args.genre_confusion_csv
        if csv_path is None:
            csv_path = args.out_dir / "genre_confusion.csv"
        write_genre_confusion_csv(csv_path, metrics.get("primary_genre") or {})

        if do_plot:
            plot_dir = args.out_dir / "plots"
            th = metrics.get("themes") or {}
            for theme, block in th.items():
                cm = block.get("confusion_matrix")
                if isinstance(cm, list) and len(cm) == 2:
                    _plot_matrix(cm, ["0", "1"], f"Theme: {theme}", plot_dir / f"theme_{theme}.png")

            pg = metrics.get("primary_genre") or {}
            ctk = pg.get("confusion_top_k") or {}
            labels = ctk.get("labels") or []
            matrix = ctk.get("matrix") or []
            if labels and matrix and len(labels) <= args.plot_max_labels:
                _plot_matrix(matrix, labels, "primary_genre (top-K + __other__)", plot_dir / "primary_genre_confusion.png")

    if not (args.quiet and args.out_dir):
        print(summary_text)


if __name__ == "__main__":
    main()
