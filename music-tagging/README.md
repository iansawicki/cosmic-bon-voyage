# Legacy music-tagging scripts

This directory holds **older workflows** tied to CSV exports, Contentful, and OpenAI Batch API. They share the same tagging **schema and prompt helpers** as the main pipeline via the [`tagging`](../tagging/) package (`import tagging.prompt`).

For current work, prefer:

- **Tagging (Vertex Gemini):** [`tagging`](../tagging/) at the repo root, or `python embed_pipeline.py tag-tracks` (see [USAGE.md](../USAGE.md)).
- **Embeddings (Voyage):** [`voyage_embed`](../voyage_embed/) and `embed_pipeline.py` embed commands.
- **Eval:** [`tagging/eval`](../tagging/eval/) — `python -m tagging.eval`.

See [EMBEDDINGS.md](EMBEDDINGS.md) for the original catalog and embedding notes.

Do not commit a Python `venv/` here; use `.gitignore` and a local virtualenv if needed.
