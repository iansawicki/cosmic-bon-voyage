"""Embed artists_ai via Voyage into voyage_ai_embedding (configurable)."""

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


def _embed_text_for_artist(a: dict[str, Any]) -> str | None:
    ct = a.get("combined_tags")
    if ct and str(ct).strip():
        return str(ct).strip()
    name = a.get("artist_name")
    if name and str(name).strip():
        return str(name).strip()
    return None


def fetch_artists(
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
        q = supabase.table("artists_ai").select("*")
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


def run_embed_artists(
    supabase: Client,
    *,
    replace_all: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    batch_size: int = 64,
    sleep_seconds: float = 0.0,
    embed_workers: int = 1,
) -> int:
    settings = get_voyage_settings()
    vector_column = get_voyage_vector_column()
    vo = make_voyage_client(settings)
    rows = fetch_artists(
        supabase, vector_column=vector_column, replace_all=replace_all, limit=limit
    )
    w = max(1, embed_workers)
    win = batch_size * w
    print(
        f"Artists to process: {len(rows)} (column={vector_column!r}, "
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
            for a in chunk:
                text = _embed_text_for_artist(a)
                if not text:
                    print(
                        f"Skip artist_name={a.get('artist_name')!r} (no combined_tags or artist_name)"
                    )
                    continue
                texts.append(text)
                valid.append(a)
            if not texts:
                continue
            if dry_run:
                print(f"[dry-run] would embed {len(texts)} artists")
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
            for a, emb in zip(valid, embs):
                update_one_or_raise(
                    supabase,
                    table="artists_ai",
                    payload={vector_column: float_vector(emb)},
                    pk_column="artist_name",
                    pk_value=a["artist_name"],
                    verify_column=vector_column,
                    pg_conn=pg,
                )
                processed += 1
            print(f"Processed {processed} / {len(rows)}")
            if sleep_seconds > 0 and i < len(rows):
                import time

                time.sleep(sleep_seconds)
    return processed
