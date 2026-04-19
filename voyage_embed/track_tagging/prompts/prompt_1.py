"""Prompt variant ``prompt_1`` (default): schema, system prompt, and row builders.

Add ``prompt_2.py``, etc., and select with env ``TAG_PROMPT`` or ``--prompt`` / ``--tag-prompt``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# JSON Schema for model output (OpenAI json_schema / Gemini response_json_schema).
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "object",
            "properties": {
                "flow": {
                    "type": "object",
                    "properties": {
                        "assigned": {"type": "boolean"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["assigned", "confidence"],
                    "additionalProperties": False,
                },
                "ritual": {
                    "type": "object",
                    "properties": {
                        "assigned": {"type": "boolean"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["assigned", "confidence"],
                    "additionalProperties": False,
                },
                "expanded_state": {
                    "type": "object",
                    "properties": {
                        "assigned": {"type": "boolean"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["assigned", "confidence"],
                    "additionalProperties": False,
                },
            },
            "required": ["flow", "ritual", "expanded_state"],
            "additionalProperties": False,
        },
        "primary_genre": {"type": "string"},
        "secondary_genres": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
        "style_tags": {"type": "array", "items": {"type": "string"}},
        "mood_keywords": {"type": "array", "items": {"type": "string"}},
        "search_keywords": {"type": "array", "items": {"type": "string"}},
        "energy_level": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": [
        "themes",
        "primary_genre",
        "secondary_genres",
        "style_tags",
        "mood_keywords",
        "search_keywords",
        "energy_level",
        "summary",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """
You are an expert music curator for a curated internet radio app centered on three core states: flow, ritual, expanded states.

These themes represent experiential states that listeners may enter while engaging with the music.

Definitions:

RITUAL:
Slow, intentional, ceremonial or repetitive practices that create a sense of sacred space, grounding, or presence. Often associated with mindful routines or reflective moments.

Examples:
sleep, meditation, tea ceremony, bathing, intimacy, journaling, prayer, walking in nature, contemplative reading, gentle movement, yoga, breath awareness.

FLOW:
A state of deep concentration and effortless engagement where attention is fully absorbed in an activity. Associated with productivity, creativity, learning, and sustained focus.

Examples:
focused work, studying, writing, coding, creative production, problem-solving, designing, reading with concentration, sustained mental effort.

EXPANDED STATES:
Altered or heightened states of consciousness where perception, awareness, or sense of self may shift beyond ordinary waking experience.

Examples:
psychedelic experiences, breathwork journeys, deep meditation, lucid dreaming, trance states, ecstatic dance, sensory immersion, deep sleep or dream states.

Given this track metadata:
- Title: {title}
- Artist: {artist}
- Album: {album}
- Playlist: {playlist_name}
- Mood description: {playlist_description}

Before assigning themes, evaluate how the track might function in a listening context.

Consider:
- listener state
- tempo and intensity
- emotional tone
- environment where the music might be used
- playlist context

Then determine the most appropriate experiential states.

Your tasks:

1. Evaluate the likelihood of each experiential state (flow, ritual, expanded_state).

For each theme determine:
- whether the state is likely to occur
- a confidence score between 0 and 1

All three themes must always be present.

Do not assign all themes equally. Prefer the one or two most strongly associated experiential states.

2. Determine the PRIMARY GENRE of the track.
This should represent the main musical family.

Examples:
Ambient, Electronic, Experimental, Downtempo, Neoclassical, Jazz, Soundtrack.

3. Determine up to TWO SECONDARY GENRES.
These should be more specific stylistic genres related to the primary genre.

Examples:
Drone, Dark Ambient, Minimal, Psychedelic Ambient, Electroacoustic.

4. Generate 5–10 STYLE TAGS describing sonic or production characteristics.

Examples:
Minimal, Hypnotic, Textural, Atmospheric, Repetitive, Layered, Organic, Field Recordings.

5. Generate 8–15 MOOD KEYWORDS describing the emotional or experiential character of the track.

Examples:
Meditative, Serene, Contemplative, Expansive, Dreamlike.

6. Generate 5–10 SEARCH KEYWORDS that help users discover the track.

7. Estimate an ENERGY LEVEL between 0.0 and 1.0 representing musical intensity.

Energy scale:
0.0 → extremely calm / ambient
0.3 → meditative / slow
0.5 → steady groove
0.7 → energetic
1.0 → intense / driving

8. Write a concise summary of the track.

Rules:

- Primary genre must be a real music genre.
- Secondary genres should refine the primary genre.
- Avoid inventing genres.
- Use standard capitalization.
- Style tags should describe sound design or structure.
- Mood keywords should describe emotional tone.

Output ONLY valid JSON matching the schema. Do not include explanations.
"""


def format_system_prompt(
    *,
    title: str,
    artist: str,
    album: str,
    playlist_name: str,
    playlist_description: str,
) -> str:
    return (
        SYSTEM_PROMPT.replace("{title}", title)
        .replace("{artist}", artist)
        .replace("{album}", album)
        .replace("{playlist_name}", playlist_name)
        .replace("{playlist_description}", playlist_description)
    )


def build_user_content(
    *,
    title: str,
    artist: str,
    album: str,
    playlist_name: str,
    playlist_description: str,
) -> str:
    return (
        f"Title: {title}\n"
        f"Artist: {artist}\n"
        f"Album: {album}\n"
        f"Playlist: {playlist_name}\n"
        f"Mood description: {playlist_description}"
    )


def _norm_word(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def build_combined_tags(parsed: dict[str, Any]) -> str:
    """Flatten genres, tags, and moods into a single lowercase space-separated string (import-compatible)."""
    parts: list[str] = []
    pg = parsed.get("primary_genre")
    if pg:
        parts.append(_norm_word(str(pg)))
    for g in parsed.get("secondary_genres") or []:
        if g:
            parts.append(_norm_word(str(g)))
    for t in parsed.get("style_tags") or []:
        if t:
            parts.append(_norm_word(str(t)))
    for t in parsed.get("mood_keywords") or []:
        if t:
            parts.append(_norm_word(str(t)))
    for t in parsed.get("search_keywords") or []:
        if t:
            parts.append(_norm_word(str(t)))
    summary = parsed.get("summary")
    if summary:
        # Light touch: add summary words (avoid duplicating everything)
        for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", str(summary)[:400]):
            lw = w.lower()
            if len(lw) > 2 and lw not in parts:
                parts.append(lw)
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return " ".join(out)


def parsed_tags_to_db_row(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map model JSON to tracks_ai update columns."""
    return {
        "primary_genre": parsed.get("primary_genre"),
        "secondary_genres": parsed.get("secondary_genres") or [],
        "style_tags": parsed.get("style_tags") or [],
        "mood_keywords": parsed.get("mood_keywords") or [],
        "search_keywords": parsed.get("search_keywords") or [],
        "themes": parsed.get("themes"),
        "energy_level": parsed.get("energy_level"),
        "summary": parsed.get("summary"),
        "combined_tags": build_combined_tags(parsed),
    }


def prompt_sha256() -> str:
    """Stable hash of the system prompt template + schema for tagging_runs."""
    payload = SYSTEM_PROMPT.encode("utf-8") + b"\n" + __import__("json").dumps(SCHEMA, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def schema_version() -> str:
    return prompt_sha256()[:16]


def openai_batch_request_body(
    *,
    title: str,
    artist: str,
    album: str,
    playlist_name: str,
    playlist_description: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> dict[str, Any]:
    """Build OpenAI chat.completions body for Batch API (legacy CSV flow)."""
    system_filled = format_system_prompt(
        title=title,
        artist=artist,
        album=album,
        playlist_name=playlist_name,
        playlist_description=playlist_description,
    )
    user_content = build_user_content(
        title=title,
        artist=artist,
        album=album,
        playlist_name=playlist_name,
        playlist_description=playlist_description,
    )
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "music_tagging_response", "strict": True, "schema": SCHEMA},
        },
        "messages": [
            {"role": "system", "content": system_filled},
            {"role": "user", "content": user_content},
        ],
    }
