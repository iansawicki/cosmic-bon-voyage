"""Load active tagging SCHEMA + helpers from ``voyage_embed.track_tagging.prompts.<name>``.

Selection: env ``TAG_PROMPT`` (default ``prompt_1``), or set env before importing
``voyage_embed.track_tagging.run``. CLI ``--prompt`` / ``--tag-prompt`` sets this before imports.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

__all__ = [
    "active_prompt_name",
    "load_prompt_module",
    "reload_prompt_module",
    "SCHEMA",
    "SYSTEM_PROMPT",
    "build_combined_tags",
    "build_user_content",
    "format_system_prompt",
    "openai_batch_request_body",
    "parsed_tags_to_db_row",
    "prompt_sha256",
    "schema_version",
]

# Re-exported from prompts.<name> via __getattr__
_DELEGATED = frozenset(
    {
        "SCHEMA",
        "SYSTEM_PROMPT",
        "build_combined_tags",
        "build_user_content",
        "format_system_prompt",
        "openai_batch_request_body",
        "parsed_tags_to_db_row",
        "prompt_sha256",
        "schema_version",
    }
)

_cached_key: str | None = None
_cached_mod: Any = None


def _normalize_name(raw: str | None) -> str:
    if not raw or not str(raw).strip():
        return "prompt_1"
    s = str(raw).strip().removesuffix(".py")
    return s or "prompt_1"


def active_prompt_name() -> str:
    return _normalize_name(os.environ.get("TAG_PROMPT"))


def load_prompt_module() -> Any:
    """Import ``prompts.<name>`` for current ``TAG_PROMPT`` (default ``prompt_1``)."""
    global _cached_key, _cached_mod
    key = active_prompt_name()
    if _cached_mod is None or key != _cached_key:
        _cached_key = key
        _cached_mod = importlib.import_module(f"voyage_embed.track_tagging.prompts.{key}")
    return _cached_mod


def reload_prompt_module() -> Any:
    """Force re-read of ``TAG_PROMPT`` and reload the prompt submodule."""
    global _cached_key, _cached_mod
    _cached_key = None
    _cached_mod = None
    m = load_prompt_module()
    importlib.reload(m)
    _cached_mod = m
    return m


def __getattr__(name: str) -> Any:
    if name in _DELEGATED:
        return getattr(load_prompt_module(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
