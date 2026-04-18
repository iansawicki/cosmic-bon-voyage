"""Natural-language search over cached embedding vectors (Voyage query embed + cosine similarity)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from voyage_embed.env import VoyageSettings, get_voyage_settings
from voyage_embed.universe_viz import UniverseModel, _as_float_vector
from voyage_embed.voyage_client import embed_texts, make_voyage_client


def stack_vectors_from_rows(
    rows: list[dict[str, Any]],
    column: str,
    max_points: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Rows with valid vectors in `column`, up to `max_points`. Same order as `tracks` embed pipeline."""
    vecs: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for row in rows[:max_points]:
        v = _as_float_vector(row.get(column))
        if v is not None:
            vecs.append(v)
            meta.append(row)
    if not vecs:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.stack(vecs, axis=0).astype(np.float32), meta


def cosine_similarities(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of unit query with each row of `vectors` (n × d)."""
    q = np.asarray(query, dtype=np.float64).ravel()
    nq = np.linalg.norm(q)
    if nq < 1e-12:
        raise ValueError("Query embedding has zero norm.")
    q = q / nq
    if vectors.size == 0:
        return np.array([], dtype=np.float64)
    vn = np.linalg.norm(vectors, axis=1, keepdims=True)
    vn = np.maximum(vn, 1e-12)
    v0 = vectors / vn
    if q.shape[0] != v0.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query has {q.shape[0]} dims, "
            f"stored vectors have {v0.shape[1]}. Use the same VOYAGE_MODEL / "
            f"VOYAGE_OUTPUT_DIMENSION as when rows were embedded."
        )
    return (v0 @ q).astype(np.float64)


def embed_query_text(text: str, settings: VoyageSettings | None = None) -> np.ndarray:
    """Embed a search string with Voyage ``input_type=query`` (retrieval-oriented)."""
    settings = settings or get_voyage_settings()
    client = make_voyage_client(settings)
    vecs = embed_texts(
        client,
        [text.strip()],
        model=settings.model,
        input_type="query",
        output_dimension=settings.output_dimension,
    )
    return np.asarray(vecs[0], dtype=np.float64)


def resolve_embedding_column(
    sample: list[dict[str, Any]],
    embedding_column: str | None,
    guess_embedding_columns: Callable[[list[str]], list[str]],
) -> str | None:
    if not sample:
        return None
    cols = guess_embedding_columns(sorted({k for row in sample for k in row.keys()}))
    return embedding_column or (cols[0] if cols else None)


def row_preview(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "track_title": str(row.get("track_title") or row.get("title") or "")[:200],
        "artist_name": str(row.get("artist_name") or row.get("artist") or "")[:200],
        "combined_tags": str(row.get("combined_tags") or "")[:400],
        "track_id": str(row.get("track_id") or row.get("id") or "")[:80],
    }


def semantic_search_rows(
    sample: list[dict[str, Any]],
    query: str,
    *,
    embedding_column: str | None,
    guess_embedding_columns: Callable[[list[str]], list[str]],
    max_points: int,
    top_k: int,
    settings: VoyageSettings | None = None,
) -> tuple[str, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns ``(column_used, scores_for_kept_rows, kept_meta_rows, top_hits)``.
    ``top_hits`` entries: ``score``, ``rank``, plus keys from ``row_preview``.
    """
    col = resolve_embedding_column(sample, embedding_column, guess_embedding_columns)
    if not col:
        raise ValueError("No embedding column: pass --embedding-column or name heuristics must match a column.")

    X, meta = stack_vectors_from_rows(sample, col, max_points)
    if X.shape[0] < 1:
        raise ValueError(f"No parseable vectors in column {col!r}.")

    qvec = embed_query_text(query, settings=settings)
    scores = cosine_similarities(qvec, X)
    order = np.argsort(-scores)
    k = min(max(1, top_k), len(order))
    top_hits: list[dict[str, Any]] = []
    for rank in range(k):
        j = int(order[rank])
        r = meta[j]
        prev = row_preview(r)
        prev["score"] = float(scores[j])
        prev["rank"] = rank + 1
        top_hits.append(prev)

    return col, scores, meta, top_hits


def similarities_for_universe_model(
    model: UniverseModel,
    query: str,
    settings: VoyageSettings | None = None,
) -> np.ndarray:
    """Cosine similarity of query to each point in an existing universe (same rows/order as model)."""
    qvec = embed_query_text(query, settings=settings)
    X, meta = stack_vectors_from_rows(model.meta_rows, model.column, len(model.meta_rows))
    if len(meta) != len(model.meta_rows):
        raise RuntimeError("Unexpected row drop in universe model.")
    return cosine_similarities(qvec, X)
