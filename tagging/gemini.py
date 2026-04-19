"""Vertex / Gemini client for structured music tagging."""

from __future__ import annotations

import json
import os
from typing import Any

from google import genai
from google.genai import types

import tagging.prompt as prompt_module


def get_genai_client() -> genai.Client:
    """
    Vertex AI: use ``GOOGLE_CLOUD_API_KEY`` (Vertex express) or
    ``GOOGLE_CLOUD_PROJECT`` + ``GOOGLE_CLOUD_LOCATION`` with ADC.

    Set ``GOOGLE_GENAI_USE_VERTEXAI=true`` to force Vertex mode when using project/location.
    """
    vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true", "yes")
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("VERTEX_LOCATION") or "us-central1"
    api_key = os.getenv("GOOGLE_CLOUD_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if vertex or project:
        if project:
            return genai.Client(vertexai=True, project=project, location=location)
        if api_key:
            return genai.Client(vertexai=True, api_key=api_key)
    if api_key:
        return genai.Client(api_key=api_key)
    raise RuntimeError(
        "Set GOOGLE_CLOUD_API_KEY (Vertex with API key), or GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION, "
        "or GOOGLE_API_KEY for Gemini API."
    )


def default_tag_model() -> str:
    return os.getenv("GEMINI_TAG_MODEL", "gemini-2.0-flash-lite")


def tag_track_sync(
    client: genai.Client,
    *,
    model: str,
    title: str,
    artist: str,
    album: str,
    playlist_name: str,
    playlist_description: str,
    temperature: float = 0.2,
    max_output_tokens: int = 8192,
) -> dict[str, Any]:
    """Call Gemini with JSON schema constrained output; return parsed dict."""
    system_text = prompt_module.format_system_prompt(
        title=title,
        artist=artist,
        album=album,
        playlist_name=playlist_name,
        playlist_description=playlist_description,
    )
    user_text = prompt_module.build_user_content(
        title=title,
        artist=artist,
        album=album,
        playlist_name=playlist_name,
        playlist_description=playlist_description,
    )

    config = types.GenerateContentConfig(
        system_instruction=system_text,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
        response_json_schema=prompt_module.SCHEMA,
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_text)],
        )
    ]

    resp = client.models.generate_content(model=model, contents=contents, config=config)
    text = (resp.text or "").strip()
    if not text:
        raise ValueError("Empty response from Gemini")
    return json.loads(text)
