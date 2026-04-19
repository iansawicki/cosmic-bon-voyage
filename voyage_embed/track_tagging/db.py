"""Supabase: fetch tracks with playlist context via track_playlist_map."""

from __future__ import annotations

from typing import Any

from supabase import Client

MAP_TABLE = "track_playlist_map"
PLAYLIST_TABLE = "playlists_ai"


def pick_playlist_id_for_track(supabase: Client, track_id: str) -> str | None:
    """Choose one playlist_id: minimum ``position`` wins."""
    r = (
        supabase.table(MAP_TABLE)
        .select("playlist_id, position")
        .eq("track_id", track_id)
        .order("position")
        .limit(1)
        .execute()
    )
    rows = r.data or []
    if not rows:
        return None
    return rows[0].get("playlist_id")


def fetch_playlist_row(supabase: Client, playlist_id: str) -> dict[str, Any] | None:
    r = supabase.table(PLAYLIST_TABLE).select("*").eq("playlist_id", playlist_id).limit(1).execute()
    rows = r.data or []
    return rows[0] if rows else None


def playlist_prompt_fields(playlist: dict[str, Any] | None) -> tuple[str, str]:
    if not playlist:
        return "", ""
    title = str(playlist.get("playlist_title") or "")
    desc = str(playlist.get("description") or "")
    tags = playlist.get("tags")
    if isinstance(tags, list) and tags:
        desc = f"{desc} Tags: {', '.join(str(t) for t in tags)}".strip()
    return title, desc


def fetch_tracks_to_tag(
    supabase: Client,
    *,
    replace_all: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    """
    Rows from tracks_ai. If not replace_all, only ``tagging_status = 'pending'`` (default for new rows).
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    page = 500
    while True:
        q = supabase.table("tracks_ai").select("*")
        if not replace_all:
            q = q.eq("tagging_status", "pending")
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


def enrich_track_with_playlist(supabase: Client, track: dict[str, Any]) -> dict[str, Any]:
    """Add playlist_name, playlist_description for prompt."""
    tid = track.get("track_id")
    if not tid:
        return {**track, "playlist_name": "", "playlist_description": ""}
    pid = pick_playlist_id_for_track(supabase, str(tid))
    if not pid:
        return {**track, "playlist_name": "", "playlist_description": ""}
    prow = fetch_playlist_row(supabase, str(pid))
    name, desc = playlist_prompt_fields(prow)
    return {**track, "playlist_name": name, "playlist_description": desc}
