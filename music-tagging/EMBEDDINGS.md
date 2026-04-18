# Music-tagging: catalog, import, and embeddings

This folder historically held **Contentful export**, **OpenAI Batch tagging**, and **embedding upload** scripts. **Embedding generation now uses Voyage AI** via shared code in the parent repo (`voyage_embed/`, `embed_pipeline.py`).

## Environment

- **Tagging / import (unchanged where applicable):** `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `OPENAI_API_KEY`, Contentful tokens as in your `.env`.
- **Voyage embeddings:** run from **bon-voyager** root and use `VOYAGE_API_KEY` and Supabase vars documented in [../USAGE.md](../USAGE.md).

## Layer A: Catalog and LLM tags (occasional)

1. Export catalog from Contentful (`export_tracks_full_catalog.py`, etc.).
2. Build `tracks_to_tag.csv` and run `tag_tracks_batch.py` (OpenAI Batch chat completions).
3. Merge outputs (`merge_batches.py`) and produce `tracks_ai_import.jsonl` (including `combined_tags`).
4. `import_tracks_supabase.py` inserts into `tracks_ai`.
5. `update_tracks_metadata.py` can enrich titles/artists from CSV.

This path does **not** call Voyage.

## Layer B: Embeddings (Voyage)

Use the parent project:

```bash
cd ..
python embed_pipeline.py embed-tracks
python embed_pipeline.py embed-artists
python embed_pipeline.py embed-playlists
```

Or invoke the thin wrappers:

```bash
python generate_embeddings.py --limit 50
python generate_artist_embeddings.py --limit 50
python generate_playlist_embeddings.py --limit 50
```

Voyage vectors are written to **`voyage_ai_embedding`** (see `VOYAGE_VECTOR_COLUMN` in [../USAGE.md](../USAGE.md)); legacy OpenAI remains in **`embedding`**.

**Text sources:**

- **tracks_ai:** `combined_tags` (fallback: `track_title` / `artist_name` / `album` fields if `combined_tags` is empty)
- **artists_ai:** `combined_tags`
- **playlists_ai:** `playlist_title`, `description`, `tags` → normalized `combined_tags` stored on update

## Model and dimensions

Default model is **`voyage-3.5-lite`** (override with `VOYAGE_MODEL`). You must align **vector column dimension** in Postgres with the Voyage output; see [../USAGE.md](../USAGE.md).

## OpenAI embeddings (legacy)

Older commits used `text-embedding-3-small`. Those scripts have been replaced; keep `OPENAI_API_KEY` only if you still run Batch tagging (`tag_tracks_batch.py`).
