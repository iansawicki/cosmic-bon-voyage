"""Offline evaluation of tag JSONL (reference vs candidate)."""

from .metrics import compute_all_metrics, merge_by_track

__all__ = ["compute_all_metrics", "merge_by_track"]
