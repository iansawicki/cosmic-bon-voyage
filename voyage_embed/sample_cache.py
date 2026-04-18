"""Persist EDA / viz sample rows to JSON to avoid repeated Supabase fetches."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_VERSION = 1

# bon-voyager repo root (parent of `voyage_embed/`)
_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve_sample_cache_path(path: str | Path, repo_root: Path | None = None) -> Path:
    """
    Resolve a cache JSON path: absolute path, then cwd-relative, then ``repo_root``-relative
    (so ``viz/foo.json`` works when the CLI is run from outside the repo).
    """
    raw = Path(path).expanduser()
    root = repo_root if repo_root is not None else _DEFAULT_REPO_ROOT
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append((Path.cwd() / raw).resolve())
        candidates.append((root / raw).resolve())
    tried = "\n  ".join(str(c) for c in candidates)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"Sample cache not found: {path!r}. Tried:\n  {tried}")


def save_sample_cache(
    path: str | Path,
    *,
    table: str,
    requested_sample: int,
    rows: list[dict[str, Any]],
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CACHE_VERSION,
        "table": table,
        "requested_sample": requested_sample,
        "row_count": len(rows),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_sample_cache(path: str | Path, repo_root: Path | None = None) -> dict[str, Any]:
    p = resolve_sample_cache_path(path, repo_root=repo_root)
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("version") != CACHE_VERSION:
        raise ValueError(
            f"Cache version mismatch (got {data.get('version')}, expected {CACHE_VERSION})"
        )
    if "rows" not in data:
        raise ValueError("Invalid cache: missing 'rows'")
    return data
