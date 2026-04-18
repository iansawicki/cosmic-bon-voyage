# Track embedding universe visualization

This repo can turn a sample of rows from Supabase (typically `tracks_ai`) into a **2D scatter plot** of embedding space: points are tracks, positions come from **t-SNE** on the chosen vector column, and **KMeans** colors clusters in the *original* embedding space (not in the 2D projection—so colors reflect semantic grouping, while layout is for exploration only).

Use this to sanity-check embeddings before building radio-style grouping or search.

## What you get

| Output | Best for | Dependencies |
|--------|----------|--------------|
| **PNG** | Quick share, slides, diff in git | `matplotlib` |
| **Plotly HTML** | Familiar controls, legend, rich hover HTML | `plotly` |
| **D3 HTML** | Pan/zoom, text filter, sidebar cluster keyword hints; optional starfield + similarity-sized points; self-contained file | D3 loaded from CDN in the HTML; no extra pip package |

All three are produced from the same computed model (t-SNE coordinates + cluster labels + metadata). Implementation lives in **`voyage_embed/universe_viz.py`**.

## Semantic search (natural language vs cached vectors)

For **analysis and visualization only**, you can rank tracks by **cosine similarity** between:

1. A **Voyage query embedding** of your text (`input_type=query`), and  
2. The **stored vectors** in the sample (same column as plots).

Requires **`VOYAGE_API_KEY`**. The query embedding must match the **model and dimension** of the stored vectors—use the same **`VOYAGE_MODEL`** and **`VOYAGE_OUTPUT_DIMENSION`** (if any) as when those rows were embedded; otherwise you get a clear dimension-mismatch error.

### `embed_pipeline.py search`

Scores rows in a cache file and prints the top hits (optional JSON).

```bash
python embed_pipeline.py search --from-cache viz/tracks_ai_sample_1500.json \
  --embedding-column voyage_ai_3p5_embed \
  "chill microdose vibes"
```

Use **`--top-k`**, **`--sample-limit`**, **`--max-points`**, **`--json-out`** as needed. The query is a **positional** argument—quote it if it contains spaces.

### `embed_pipeline.py viz --search`

- With **`--plotly`** and/or **`--d3`**: runs the query through Voyage, prints the top **`--search-top`** hits (default 15), and colors **Plotly** and **D3** points by **cosine similarity** (Viridis). PNG stays cluster-colored.
- **Search only** (no plots): omit `--plotly` / `--png` / `--d3` and pass **`--search "..."`** to print ranked hits without running t-SNE (faster).

```bash
python embed_pipeline.py viz --from-cache viz/tracks_ai_sample_1500.json \
  --embedding-column voyage_ai_3p5_embed \
  --d3 tracks_query_d3.html --plotly tracks_query.html \
  --search "chill microdose vibes"
```

Implementation: **`voyage_embed/semantic_search.py`**.

### Interactive browser search (`embed_pipeline.py cosmos`)

Static D3/Plotly files cannot call Voyage from the browser without exposing your API key. For **live** semantic search in the UI, run a small local server: it keeps **`VOYAGE_API_KEY`** on the machine, embeds each new query server-side, and returns cosine similarities to the page.

```bash
python embed_pipeline.py cosmos --cache viz/tracks_sample_1500.json \
  --embedding-column voyage_ai_3p5_embed
```

Open **http://127.0.0.1:8765/** (default). Use the **Voyage** search row in the header, then **Search** — markers recolor by similarity. The D3 view also supports **constellation** (2D kNN links), **sector** labels at cluster centroids, **signal calibration** (min/median/max + histogram), **hyperspace zoom** toward the high-match region after a search, and a **mission log** (recent queries in `localStorage`). Cache paths are resolved from your **current directory** and from the **repo root**, so `viz/tracks_sample_1500.json` works even when the filename differs from older docs.

Or: `python -m voyage_embed.cosmos_server --cache ...` (same flags).

## Prerequisites

```bash
pip install scikit-learn matplotlib plotly
```

- **Supabase**: Same env as the rest of the project (`SUPABASE_URL` / `SUPABASE_URL_DEV`, service role key). See [USAGE.md](USAGE.md). Search/viz-from-cache does not call Supabase once the JSON exists.
- **Embedding column**: A column of float vectors (e.g. `voyage_ai_3p5_embed`). If omitted, the EDA script guesses columns whose names look like embeddings (`embedding`, `voyage`, `vector`, etc.).

## Commands

### EDA + plots (`embed_pipeline.py eda`)

Wraps `sbase_embeddings_rerun.py`. Useful flags:

| Flag | Meaning |
|------|---------|
| `--sample N` | Max rows to pull (paginated from PostgREST). |
| `--embedding-column NAME` | Vector column for t-SNE. |
| `--plot-universe PATH` | `.html` → Plotly; `.png` → matplotlib. |
| `--d3-universe PATH` | Additional D3 HTML output. |
| `--plot-max-points N` | Cap rows fed to t-SNE (default in script is 3000). |
| `--plot-clusters K` | Fixed KMeans `K`; `0` = automatic. |
| `--sample-cache PATH` | Read/write JSON cache of sampled rows. |
| `--refresh-cache` | Refetch from Supabase and overwrite the cache file. |

**Example** (Plotly + D3 + cache):

```bash
mkdir -p viz
python embed_pipeline.py eda --sample 1500 \
  --sample-cache viz/tracks_ai_sample_1500.json \
  --plot-universe tracks_universe.html \
  --d3-universe tracks_universe_d3.html \
  --embedding-column voyage_ai_3p5_embed
```

You can call `python sbase_embeddings_rerun.py` with the same flags if you prefer not to use the orchestrator.

### Render from cache only (`embed_pipeline.py viz`)

Regenerates HTML/PNG from a **`--sample-cache`** JSON file **without** fetching rows from Supabase. Still runs **t-SNE and KMeans locally** (needs `scikit-learn`; Plotly/matplotlib only if you ask for those outputs).

| Flag | Meaning |
|------|---------|
| `--from-cache PATH` | Required. JSON written by EDA `--sample-cache`. |
| `--embedding-column` | Optional; same heuristic as EDA if omitted. |
| `--sample-limit N` | Use only the first `N` cached rows (`0` = all). |
| `--plot-max-points`, `--plot-clusters` | Same idea as EDA. |
| `--plotly`, `--png`, `--d3` | At least one of these **or** `--search` (see Semantic search above). |
| `--search TEXT` | Voyage query string; optional similarity coloring for Plotly/D3. |
| `--search-top N` | How many hits to print with `--search` (default 15). |

**Example**:

```bash
python embed_pipeline.py viz --from-cache viz/tracks_ai_sample_1500.json \
  --embedding-column voyage_ai_3p5_embed \
  --plotly tracks_universe.html --d3 tracks_universe_d3.html
```

## Caching behavior

- First run **with** `--sample-cache PATH`: fetches up to `--sample` rows, writes JSON (versioned schema in `voyage_embed/sample_cache.py`), then runs plots.
- Later runs: if the file exists, the table name matches, and the cache has **at least** `--sample` rows, the sample is loaded from disk and the **paginated table fetch is skipped**. Row count and column profiling still use the loaded sample.
- **`--refresh-cache`**: Always refetch and overwrite the cache.
- If you increase `--sample` above what the cache was built for, the tool refetches automatically.

## Interpreting the UI

- **Terminal**: After clustering, **cluster hints** print token frequencies from metadata (`combined_tags`, title, artist, album, etc.)—handy for naming clusters mentally.
- **Plotly**: Hover shows escaped HTML for title, artist, tags, id.
- **D3**: Metadata search box filters which dots stay bright; sidebar lists top terms per cluster; zoom/pan with mouse. With **`--search`**, marker color encodes **cosine similarity** to the Voyage query, and the sidebar shows the query text.

## Troubleshooting

- **“Not enough parsed vectors … (need >= 5)”** — Column missing, wrong type, or mostly null; check `--embedding-column` and a few rows in Supabase.
- **PostgREST timeout** — Lower `--sample` or increase server/statement limits.
- **Cache version error** — `CACHE_VERSION` in `sample_cache.py` changed; delete the old JSON or use `--refresh-cache`.
- **Embedding dimension mismatch** (search) — Align `VOYAGE_MODEL` / `VOYAGE_OUTPUT_DIMENSION` with how the table column was produced.

## Related code

- `voyage_embed/universe_viz.py` — t-SNE, KMeans, Plotly, matplotlib, D3 HTML template.
- `voyage_embed/semantic_search.py` — Voyage query embed + cosine ranking vs cached vectors.
- `voyage_embed/sample_cache.py` — JSON cache format.
- `sbase_embeddings_rerun.py` — EDA, fetch, cache load/save, delegates plotting to `universe_viz`.
- `embed_pipeline.py` — `eda`, `viz`, `search`, and `cosmos` subcommands.
- `voyage_embed/cosmos_server.py` — local HTTP + `/api/search` (Voyage), `/api/sign-audio` (JSON map from Edge), and **`/api/stream-audio?assetName=…`** (same-origin byte proxy to CloudFront with `Range` forwarding — use this for playback so the browser is not blocked by CDN CORS). Double-click a dot when `audio_key` is present. Requires `SUPABASE_URL` / `SUPABASE_URL_DEV` and `SUPABASE_ANON_KEY` or `SUPABASE_KEY`.
- If streaming returns **403** after path-encoding fixes, redeploy cosmic-app **CloudFront signers** so signed URLs use percent-encoded paths (matching the proxy). **Clear or wait out** `signed_url_cache` rows that were signed with raw spaces in the URL.
- `voyage_embed/cosmic_stream.py` — HTTP helper for the batch signer; guesses `audio/*` MIME when CDN sends `application/octet-stream` (fixes Chrome “no supported source”).
- **Galactic tripper** (cosmos + Voyage search): after a successful semantic search, the sidebar shows a **12-stop** trajectory (top matches by similarity, order greedy in 2D), a **purple path** on the plot, and dimmed non-trip dots. Controls:
    - **Take flight** — streams the trip as a playlist through the audio dock. The token **glides between consecutive stops** during each track (position = `currentTime / duration`), the active stop dot glows, the sidebar highlights the current item, and the next track auto-loads on `ended`. Unsupported formats (e.g. FLAC in Safari) and 403s are skipped automatically; three consecutive failures abort the flight. **Skip stop** jumps early; **Stop flight** cancels. Manually double-clicking a different dot also cancels the flight and plays that track.
    - **Preview** — camera frames the trip and animates a token along the full path once (no audio). Good for a quick overview.
    - **Clear** — removes the path and returns the field to its normal state.

For the broader embedding pipeline (backfill, env), see [USAGE.md](USAGE.md).
