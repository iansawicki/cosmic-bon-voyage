"""
Explore `tracks_ai` (or another table) before re-embedding with Voyage.

- Row counts, column inventory, null rates, and value patterns
- Heuristic flags for text you might concatenate for embedding models
- Optional 2D preview plot when a float vector column already exists (old embeddings)

After you upload Voyage vectors, you can build radio-style groups the same way as
`~/Dev/music_flow`: kNN graph + clusters (`flow/graph.py`) and HTML viz (`flow/visualize_graph.py`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from supabase import Client

from voyage_embed.env import get_supabase_client
from voyage_embed.sample_cache import load_sample_cache, save_sample_cache
from voyage_embed.universe_viz import (
    UniverseModel,
    compute_universe,
    write_d3_universe,
    write_matplotlib_universe,
    write_plotly_universe,
    _hover_html_for_row,
    cluster_hints,
)

_REPO_ROOT = Path(__file__).resolve().parent

load_dotenv(_REPO_ROOT / ".env")

# Columns whose *names* suggest good Voyage input (concatenate non-empty values).
TEXT_SIGNAL_NAMES = (
    "title",
    "name",
    "track_title",
    "song",
    "artist",
    "artists",
    "album",
    "album_name",
    "genre",
    "genres",
    "mood",
    "tags",
    "label",
    "year",
    "bpm",
    "key",
    "description",
    "lyrics_snippet",
)

# Names that usually hold vectors / old embeddings (for profiling & optional plot).
EMBEDDING_NAME_HINTS = (
    "embedding",
    "embeddings",
    "vector",
    "vectors",
    "voyage",
    "openai",
    "ada",
    "clip",
)


def _supabase() -> Client:
    return get_supabase_client()


def _fetch_count(client: Client, table: str) -> int | None:
    r = client.table(table).select("*", count="exact").limit(1).execute()
    return getattr(r, "count", None)


def _fetch_sample(client: Client, table: str, limit: int) -> list[dict[str, Any]]:
    """Paginate PostgREST range requests (default page size is often 1000)."""
    page = 1000
    rows: list[dict[str, Any]] = []
    offset = 0
    while len(rows) < limit:
        end = offset + page - 1
        r = client.table(table).select("*").range(offset, end).execute()
        batch = r.data or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return rows[:limit]


def _as_float_vector(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        v = val.astype(np.float64, copy=False).ravel()
        return v if v.size else None
    if isinstance(val, (list, tuple)):
        try:
            v = np.asarray(val, dtype=np.float64).ravel()
            return v if v.size else None
        except (ValueError, TypeError):
            return None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                v = np.array(json.loads(s), dtype=np.float64).ravel()
                return v if v.size else None
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def _type_label(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int) and not isinstance(v, bool):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return f"list(len={len(v)})"
    if isinstance(v, dict):
        return f"dict(keys={len(v)})"
    return type(v).__name__


def _profile_column(name: str, values: list[Any]) -> dict[str, Any]:
    n = len(values)
    nulls = sum(1 for v in values if v is None)
    non_null = [v for v in values if v is not None]
    types = Counter(_type_label(v) for v in non_null)
    out: dict[str, Any] = {
        "column": name,
        "n": n,
        "null_pct": round(100.0 * nulls / n, 2) if n else 0.0,
        "types_seen": dict(types),
    }
    if not non_null:
        return out

    strs = [v for v in non_null if isinstance(v, str)]
    if strs:
        lens = [len(s) for s in strs]
        out["str_len_min"] = min(lens)
        out["str_len_max"] = max(lens)
        out["str_len_mean"] = round(sum(lens) / len(lens), 1)
        uniq = len(set(s.strip() for s in strs if s.strip()))
        out["distinct_str_approx"] = uniq
        samples = []
        seen: set[str] = set()
        for s in strs:
            t = s.strip()
            if t and t not in seen and len(samples) < 3:
                seen.add(t)
                samples.append(t[:120] + ("…" if len(t) > 120 else ""))
        out["examples"] = samples

    nums = [v for v in non_null if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums:
        out["num_min"] = min(nums)
        out["num_max"] = max(nums)

    vecs = [_as_float_vector(v) for v in non_null]
    vecs = [v for v in vecs if v is not None]
    if vecs:
        dims = [v.size for v in vecs]
        out["vector_dim_min"] = int(min(dims))
        out["vector_dim_max"] = int(max(dims))
        if len(set(dims)) == 1:
            norms = [float(np.linalg.norm(v)) for v in vecs[:500]]
            out["vector_l2_norm_mean"] = round(sum(norms) / len(norms), 4)
            out["vector_l2_norm_min"] = round(min(norms), 4)
            out["vector_l2_norm_max"] = round(max(norms), 4)

    return out


def _guess_text_columns(columns: list[str]) -> list[str]:
    lower = {c.lower(): c for c in columns}
    picked: list[str] = []
    for hint in TEXT_SIGNAL_NAMES:
        if hint in lower:
            picked.append(lower[hint])
    # De-dupe preserve order
    seen: set[str] = set()
    return [c for c in picked if c not in seen and not seen.add(c)]


def _guess_embedding_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for c in columns:
        cl = c.lower()
        if any(h in cl for h in EMBEDDING_NAME_HINTS):
            out.append(c)
    return out


def _build_sample_embedding_text(row: dict[str, Any], text_cols: list[str]) -> str:
    parts: list[str] = []
    for c in text_cols:
        v = row.get(c)
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, list):
            parts.append(" ".join(str(x) for x in v if x is not None))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(f"{c}: {v}")
    return " | ".join(parts)


def _print_report(
    *,
    table: str,
    total_rows: int | None,
    sample: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    text_cols: list[str],
    emb_candidates: list[str],
) -> None:
    print("=" * 72)
    print(f"Table: {table}")
    if total_rows is not None:
        print(f"Total rows (exact): {total_rows}")
    print(f"Sample size: {len(sample)}")
    print()

    if not sample:
        print("No rows returned — check table name, RLS policies, and credentials.")
        return

    cols = sorted({k for row in sample for k in row.keys()})
    print("Columns in sample:", ", ".join(cols))
    print()
    print("--- Column profiles (sample-based) ---")
    for p in profiles:
        line = f"{p['column']}: null {p['null_pct']}%  types {p.get('types_seen', {})}"
        extras = []
        if "str_len_mean" in p:
            extras.append(f"str_len ~{p['str_len_mean']}")
        if "vector_dim_min" in p:
            extras.append(f"vector_dim {p.get('vector_dim_min')}-{p.get('vector_dim_max')}")
        if extras:
            line += "  |  " + "  ".join(extras)
        print(line)
        if p.get("examples"):
            for ex in p["examples"]:
                print(f"    e.g. {ex!r}")
    print()

    print("--- Heuristic: text fields for embedding models ---")
    if text_cols:
        print("Name-matched columns to concatenate (edit TEXT_SIGNAL_NAMES in script if needed):")
        print(" ", ", ".join(text_cols))
        print()
        print("Three sample lines you might send to Voyage (metadata + text):")
        for i, row in enumerate(sample[:3]):
            line = _build_sample_embedding_text(row, text_cols)
            print(f"  [{i}] {line[:400]}{'…' if len(line) > 400 else ''}")
    else:
        print("No columns matched TEXT_SIGNAL_NAMES — inspect column profiles above and add names.")
    print()

    print("--- Heuristic: possible existing vector / embedding columns ---")
    if emb_candidates:
        print(" ", ", ".join(emb_candidates))
    else:
        print("  (none by name — check profiles for list/float arrays)")
    print()
    print("Next: point Voyage at a single string per track; keep id + source columns for joins.")
    print("Radio grouping: after vectors exist, mirror music_flow kNN + cluster → station seeds.")


def _print_cluster_hints_model(model: UniverseModel) -> None:
    print(
        "\n--- Cluster hints (token counts from metadata; KMeans in embedding space, not t-SNE) ---"
    )
    for h in cluster_hints(model.labels, model.meta_rows):
        terms = h["terms"]
        print(f"  Cluster {h['id']} (n={h['n']}): {', '.join(terms) if terms else '(no text)'}")


def _maybe_plot_universe(
    sample: list[dict[str, Any]],
    column: str | None,
    out_path: str,
    max_points: int,
    num_clusters: int = 0,
    d3_path: str = "",
) -> None:
    if not sample or (not out_path and not d3_path):
        return

    model = compute_universe(sample, column, max_points, num_clusters, _guess_embedding_columns)
    if model is None:
        cols = _guess_embedding_columns(sorted({k for row in sample for k in row.keys()}))
        col = column or (cols[0] if cols else None)
        if not col:
            print("No --embedding-column and no embedding-like column name found; skip plot.")
            return
        print(f"Not enough parsed vectors in column {col!r} for t-SNE (need >= 5). Skip plot.")
        return

    _print_cluster_hints_model(model)
    hover_texts = [_hover_html_for_row(r) for r in model.meta_rows]
    n = model.xy.shape[0]
    col = model.column

    if out_path:
        lower = out_path.lower()
        if lower.endswith(".html"):
            write_plotly_universe(out_path, model, hover_texts)
            print(f"Wrote interactive universe plot: {out_path} (column={col!r}, n={n})")
        elif lower.endswith(".png"):
            write_matplotlib_universe(out_path, model)
            print(f"Wrote universe preview plot: {out_path} (column={col!r}, n={n})")
        else:
            print(f"Unrecognized --plot-universe extension (use .html or .png): {out_path!r}")

    if d3_path:
        write_d3_universe(d3_path, model)
        print(f"Wrote D3 universe: {d3_path} (column={col!r}, n={n})")


def _try_load_sample_cache(cache_path: str, table: str, sample: int) -> list[dict[str, Any]] | None:
    try:
        data = load_sample_cache(cache_path, repo_root=_REPO_ROOT)
    except FileNotFoundError:
        return None
    except ValueError as e:
        raise SystemExit(str(e)) from e
    if data.get("table") != table:
        print(
            f"Sample cache table mismatch ({data.get('table')!r} vs {table!r}); fetching fresh sample."
        )
        return None
    rows: list[dict[str, Any]] = data["rows"]
    if len(rows) < sample:
        print(
            f"Sample cache has only {len(rows)} rows; need at least --sample={sample}. Fetching fresh sample."
        )
        return None
    return rows[:sample]


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for Supabase tracks_ai before Voyage embeddings.")
    parser.add_argument("--table", default="tracks_ai", help="Postgres table name (public schema)")
    parser.add_argument("--sample", type=int, default=2000, help="Max rows to pull for profiling")
    parser.add_argument("--json-out", default="", help="Write full report JSON to this path")
    parser.add_argument(
        "--plot-universe",
        default="",
        help="Path for t-SNE output: .html = interactive Plotly; .png = static image",
    )
    parser.add_argument(
        "--embedding-column",
        default="",
        help="Column with float vector for --plot-universe; default: auto-detect by name",
    )
    parser.add_argument("--plot-max-points", type=int, default=3000, help="Cap rows for TSNE")
    parser.add_argument(
        "--plot-clusters",
        type=int,
        default=0,
        metavar="K",
        help="KMeans clusters for coloring (0 = auto from sample size).",
    )
    parser.add_argument(
        "--sample-cache",
        default="",
        help="JSON path to load/save sampled rows (skip Supabase fetch when fresh enough).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore existing --sample-cache and refetch from Supabase.",
    )
    parser.add_argument(
        "--d3-universe",
        default="",
        help="Additional self-contained D3 HTML (zoom/pan/search); independent of --plot-universe format.",
    )
    args = parser.parse_args()

    client = _supabase()
    total = _fetch_count(client, args.table)

    sample: list[dict[str, Any]] | None = None
    if args.sample_cache and not args.refresh_cache:
        sample = _try_load_sample_cache(args.sample_cache, args.table, args.sample)
        if sample is not None:
            print(f"Using sample cache: {args.sample_cache} ({len(sample)} rows)\n")

    if sample is None:
        sample = _fetch_sample(client, args.table, args.sample)
        if args.sample_cache:
            save_sample_cache(
                args.sample_cache,
                table=args.table,
                requested_sample=args.sample,
                rows=sample,
            )
            print(f"Saved sample cache: {args.sample_cache} ({len(sample)} rows)\n")

    columns = sorted({k for row in sample for k in row.keys()}) if sample else []
    profiles = []
    for c in columns:
        values = [row.get(c) for row in sample]
        profiles.append(_profile_column(c, values))

    text_cols = _guess_text_columns(columns)
    emb_candidates = _guess_embedding_columns(columns)

    _print_report(
        table=args.table,
        total_rows=total,
        sample=sample,
        profiles=profiles,
        text_cols=text_cols,
        emb_candidates=emb_candidates,
    )

    report = {
        "table": args.table,
        "total_rows_exact": total,
        "sample_size": len(sample),
        "columns": columns,
        "column_profiles": profiles,
        "heuristic_text_columns": text_cols,
        "heuristic_embedding_columns": emb_candidates,
    }
    if args.json_out:
        p = Path(args.json_out)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"Wrote JSON report: {p}")

    if args.plot_universe or args.d3_universe:
        _maybe_plot_universe(
            sample,
            args.embedding_column or None,
            args.plot_universe,
            max_points=args.plot_max_points,
            num_clusters=max(0, args.plot_clusters),
            d3_path=args.d3_universe or "",
        )


if __name__ == "__main__":
    main()
