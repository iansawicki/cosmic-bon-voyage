"""Embed playlists_ai via Voyage into voyage_ai_embedding (configurable)."""

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


def _playlist_text(row: dict[str, Any]) -> str:
    title = row.get("playlist_title") or ""
    description = row.get("description") or ""
    tags = row.get("tags") or []
    tags_text = " ".join(tags) if isinstance(tags, list) else str(tags)
    combined = f"{title} {description} {tags_text}".lower().strip()
    return combined


def fetch_playlists(
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
        q = supabase.table("playlists_ai").select("*")
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


def run_embed_playlists(
    supabase: Client,
    *,
    replace_all: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    batch_size: int = 32,
    sleep_seconds: float = 0.0,
    embed_workers: int = 1,
) -> int:
    settings = get_voyage_settings()
    vector_column = get_voyage_vector_column()
    vo = make_voyage_client(settings)
    rows = fetch_playlists(
        supabase, vector_column=vector_column, replace_all=replace_all, limit=limit
    )
    w = max(1, embed_workers)
    win = batch_size * w
    print(
        f"Playlists to process: {len(rows)} (column={vector_column!r}, "
        f"batch_size={batch_size}, embed_workers={w}, row_window={win})"
    )
    if not rows:
        return 0

    work: list[tuple[dict[str, Any], str]] = []
    for p in rows:
        pid = p.get("playlist_id")
        combined = _playlist_text(p)
        if not combined:
            print(f"Skip playlist_id={pid!r} (no text)")
            continue
        work.append((p, combined))

    processed = 0
    logged_dim = False
    i = 0
    with optional_pg_connection() as pg:
        if pg:
            print(
                "Writes: direct Postgres (DATABASE_URL); reads/fetch still use Supabase API."
            )
        while i < len(work):
            chunk = work[i : i + win]
            i += len(chunk)
            texts = [c for _, c in chunk]
            plist_rows = [r for r, _ in chunk]
            if dry_run:
                print(f"[dry-run] would embed {len(texts)} playlists")
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
            for p, combined_tags, emb in zip(plist_rows, texts, embs):
                update_one_or_raise(
                    supabase,
                    table="playlists_ai",
                    payload={
                        vector_column: float_vector(emb),
                        "combined_tags": combined_tags,
                    },
                    pk_column="playlist_id",
                    pk_value=p["playlist_id"],
                    verify_column=vector_column,
                    pg_conn=pg,
                )
                processed += 1
            print(f"Processed {processed} / {len(work)}")
            if sleep_seconds > 0 and i < len(work):
                import time

                time.sleep(sleep_seconds)
    return processed
