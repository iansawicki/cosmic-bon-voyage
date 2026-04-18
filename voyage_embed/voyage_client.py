"""Voyage AI embed helper."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import voyageai

from voyage_embed.env import VoyageSettings, get_voyage_settings


def make_voyage_client(settings: VoyageSettings | None = None) -> voyageai.Client:
    settings = settings or get_voyage_settings()
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set VOYAGE_API_KEY (see https://docs.voyageai.com/docs/api-key-and-installation)."
        )
    return voyageai.Client(
        api_key=api_key,
        max_retries=settings.max_retries,
        timeout=settings.timeout,
    )


def embed_texts(
    client: voyageai.Client,
    texts: list[str],
    *,
    model: str,
    input_type: str = "document",
    output_dimension: int | None = None,
    truncation: bool = True,
) -> list[list[float]]:
    """Embed a list of strings in one API call (order preserved)."""
    if not texts:
        return []
    kwargs: dict[str, Any] = {
        "texts": texts,
        "model": model,
        "input_type": input_type,
        "truncation": truncation,
    }
    if output_dimension is not None:
        kwargs["output_dimension"] = output_dimension
    result = client.embed(**kwargs)
    return result.embeddings


def embed_texts_parallel_chunks(
    client: voyageai.Client,
    texts: list[str],
    *,
    model: str,
    input_type: str = "document",
    output_dimension: int | None = None,
    truncation: bool = True,
    batch_size: int = 64,
    max_workers: int = 1,
) -> list[list[float]]:
    """
    Split ``texts`` into chunks of ``batch_size`` and call ``embed`` per chunk.
    When ``max_workers`` > 1, run up to ``min(max_workers, num_chunks)`` chunk
    requests concurrently (thread pool). Order of returned embeddings matches ``texts``.
    """
    if not texts:
        return []
    chunks: list[list[str]] = []
    for j in range(0, len(texts), batch_size):
        chunks.append(texts[j : j + batch_size])

    if max_workers <= 1 or len(chunks) <= 1:
        out: list[list[float]] = []
        for c in chunks:
            out.extend(
                embed_texts(
                    client,
                    c,
                    model=model,
                    input_type=input_type,
                    output_dimension=output_dimension,
                    truncation=truncation,
                )
            )
        return out

    def run_chunk(idx: int, sub: list[str]) -> tuple[int, list[list[float]]]:
        return idx, embed_texts(
            client,
            sub,
            model=model,
            input_type=input_type,
            output_dimension=output_dimension,
            truncation=truncation,
        )

    n_workers = min(max_workers, len(chunks))
    ordered: list[list[list[float]] | None] = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        pending = [ex.submit(run_chunk, i, c) for i, c in enumerate(chunks)]
        for fut in as_completed(pending):
            idx, vecs = fut.result()
            ordered[idx] = vecs
    flat: list[list[float]] = []
    for part in ordered:
        assert part is not None
        flat.extend(part)
    return flat


def embed_texts_batched(
    client: voyageai.Client,
    texts: list[str],
    *,
    model: str,
    input_type: str = "document",
    output_dimension: int | None = None,
    batch_size: int = 64,
    sleep_seconds: float = 0.0,
) -> list[list[float]]:
    """Embed many strings, chunking API calls. Returns flat list aligned with `texts`."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        out.extend(
            embed_texts(
                client,
                chunk,
                model=model,
                input_type=input_type,
                output_dimension=output_dimension,
            )
        )
        if sleep_seconds > 0 and i + batch_size < len(texts):
            time.sleep(sleep_seconds)
    return out
