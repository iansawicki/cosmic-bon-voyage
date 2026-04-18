"""Embed tracks_ai via Voyage into voyage_ai_embedding (configurable), leaving OpenAI `embedding` intact."""

from __future__ import annotations

from typing import Any

from supabase import Client

from voyage_embed.db_update import (
    float_vector,
    optional_pg_connection,
    update_one_or_raise,
)
from voyage_embed.env import get_voyage_settings, get_voyage_vector_column
from voyage_embed.voyage_client import embed_texts_parallel_chunks, make_voyage_client


def _embed_text_for_track(t: dict[str, Any]) -> str | None:
    """Prefer combined_tags; fall back to title/artist/album if missing."""
    ct = t.get("combined_tags")
    if ct and str(ct).strip():
        return str(ct).strip()
    parts = []
    for key in ("track_title", "title", "artist_name", "artist", "album_name", "album"):
        v = t.get(key)
        if v and str(v).strip():
            parts.append(str(v).strip())
    if parts:
        return " | ".join(parts)
    return None


def fetch_tracks(
    supabase: Client,
    *,
    vector_column: str,
    replace_all: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page = 1000
    while True:
        q = supabase.table("tracks_ai").select("*")
        if not replace_all:
            q = q.is_(vector_column, "null")
        r = q.range(offset, offset + page - 1).execute()
        batch = r.data or []
        if not batch:
            break
        rows.extend(batch)
        if limit is not None and len(rows) >= limit:
            return rows[:limit]
        if len(batch) < page:
            break
        offset += page
    return rows if limit is None else rows[: limit]


def run_embed_tracks(
    supabase: Client,
    *,
    replace_all: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    batch_size: int = 64,
    sleep_seconds: float = 0.0,
    embed_workers: int = 1,
) -> int:
    """
    Returns number of tracks processed (embeddings written or dry-run counted).
    """
    settings = get_voyage_settings()
    vector_column = get_voyage_vector_column()
    vo = make_voyage_client(settings)
    rows = fetch_tracks(
        supabase, vector_column=vector_column, replace_all=replace_all, limit=limit
    )
    w = max(1, embed_workers)
    win = batch_size * w
    print(
        f"Tracks to process: {len(rows)} (replace_all={replace_all}, column={vector_column!r}, "
        f"batch_size={batch_size}, embed_workers={w}, row_window={win})"
    )
    if not rows:
        return 0

    processed = 0
    logged_dim = False
    i = 0
    with optional_pg_connection() as pg:
        if pg:
            print(
                "Writes: direct Postgres (DATABASE_URL); reads/fetch still use Supabase API."
            )
        while i < len(rows):
            chunk = rows[i : i + win]
            i += len(chunk)
            texts: list[str] = []
            valid: list[dict[str, Any]] = []
            for t in chunk:
                text = _embed_text_for_track(t)
                if not text:
                    print(
                        f"Skip track_id={t.get('track_id')!r} (no combined_tags and no title/artist fallback)"
                    )
                    continue
                texts.append(text)
                valid.append(t)
            if not texts:
                continue
            if dry_run:
                print(
                    f"[dry-run] would embed {len(texts)} tracks in this batch (e.g. track_id={valid[0].get('track_id')})"
                )
                processed += len(texts)
                continue

            embs = embed_texts_parallel_chunks(
                vo,
                texts,
                model=settings.model,
                input_type=settings.input_type,
                output_dimension=settings.output_dimension,
                batch_size=batch_size,
                max_workers=w,
            )
            if embs and len(embs[0]) > 0 and not logged_dim:
                print(f"  embedding dim={len(embs[0])}")
                logged_dim = True
            for t, emb in zip(valid, embs):
                update_one_or_raise(
                    supabase,
                    table="tracks_ai",
                    payload={vector_column: float_vector(emb)},
                    pk_column="track_id",
                    pk_value=t["track_id"],
                    verify_column=vector_column,
                    context="If PATCH failed with a HTTP error, check vector(N) matches Voyage output dim.",
                    pg_conn=pg,
                )
                processed += 1
            print(f"Processed {processed} / {len(rows)}")
            if sleep_seconds > 0 and i < len(rows):
                import time

                time.sleep(sleep_seconds)
    return processed
