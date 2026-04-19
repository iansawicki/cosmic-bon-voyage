"""Run Vertex Gemini tagging over tracks_ai."""

from __future__ import annotations

import os
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from supabase import Client

import tagging.prompt as prompt_module

from tagging.db import enrich_track_with_playlist, fetch_tracks_to_tag
from tagging.gemini import default_tag_model, get_genai_client, tag_track_sync


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).resolve().parents[1]),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def run_tag_tracks(
    supabase: Client,
    *,
    replace_all: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    workers: int = 1,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    Create a tagging_runs row, tag each track, update tracks_ai.

    Requires table ``tagging_runs`` and column ``tracks_ai.tagging_run_id`` (migration).
    """
    model = model or default_tag_model()
    use_thread_local = not dry_run and workers > 1
    client = None if dry_run else (None if use_thread_local else get_genai_client())
    thread_local = threading.local()

    def resolve_client() -> Any:
        if dry_run:
            return None
        if use_thread_local:
            if not hasattr(thread_local, "gemini"):
                thread_local.gemini = get_genai_client()
            return thread_local.gemini
        return client

    rows = fetch_tracks_to_tag(supabase, replace_all=replace_all, limit=limit)
    for i, t in enumerate(rows):
        rows[i] = enrich_track_with_playlist(supabase, t)

    if not rows:
        return {"tagging_run_id": None, "processed": 0, "ok": 0, "errors": [], "model": model, "message": "no rows to tag"}

    run_id = str(uuid.uuid4())
    sha = prompt_module.prompt_sha256()
    ver = prompt_module.schema_version()
    git = _git_commit()

    if not dry_run:
        supabase.table("tagging_runs").insert(
            {
                "id": run_id,
                "inference_provider": "vertex_gemini",
                "model": model,
                "prompt_sha256": sha,
                "schema_version": ver,
                "git_commit": git,
                "notes": {
                    "replace_all": replace_all,
                    "limit": limit,
                    "workers": workers,
                    "prompt": prompt_module.active_prompt_name(),
                },
                "batch_status": "in_progress",
            }
        ).execute()

    ok = 0
    err: list[str] = []
    db_lock = threading.Lock()

    def one(track: dict[str, Any]) -> tuple[str, bool, str | None]:
        tid = str(track.get("track_id") or "")
        title = str(track.get("track_title") or track.get("title") or "")
        artist = str(track.get("artist_name") or track.get("artist") or "")
        album = str(track.get("album_name") or track.get("album") or "")
        pl_name = str(track.get("playlist_name") or "")
        pl_desc = str(track.get("playlist_description") or "")
        if dry_run:
            return tid, True, None
        cl = resolve_client()
        assert cl is not None
        try:
            parsed = tag_track_sync(
                cl,
                model=model,
                title=title,
                artist=artist,
                album=album,
                playlist_name=pl_name,
                playlist_description=pl_desc,
                temperature=temperature,
            )
            update = prompt_module.parsed_tags_to_db_row(parsed)
            update["tagging_status"] = "tagged"
            update["tagging_run_id"] = run_id
            with db_lock:
                supabase.table("tracks_ai").update(update).eq("track_id", tid).execute()
            return tid, True, None
        except Exception as e:  # noqa: BLE001
            return tid, False, str(e)

    if workers <= 1:
        for t in rows:
            tid, success, msg = one(t)
            if success:
                ok += 1
            elif msg:
                err.append(f"{tid}: {msg}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(one, t): t for t in rows}
            for fut in as_completed(futs):
                tid, success, msg = fut.result()
                if success:
                    ok += 1
                elif msg:
                    err.append(f"{tid}: {msg}")

    if not dry_run:
        supabase.table("tagging_runs").update(
            {"batch_status": "completed", "notes": {"ok": ok, "errors": err[:50], "total": len(rows)}}
        ).eq("id", run_id).execute()

    return {"tagging_run_id": run_id, "processed": len(rows), "ok": ok, "errors": err, "model": model}
