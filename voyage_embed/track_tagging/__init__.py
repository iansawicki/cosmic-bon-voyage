"""Vertex Gemini tagging for tracks_ai (shared prompt/schema with legacy OpenAI batch)."""

from voyage_embed.track_tagging.prompt import SCHEMA, SYSTEM_PROMPT, build_combined_tags, parsed_tags_to_db_row

__all__ = [
    "SCHEMA",
    "SYSTEM_PROMPT",
    "build_combined_tags",
    "parsed_tags_to_db_row",
]
