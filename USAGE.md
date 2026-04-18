# Bon Voyage: Supabase dev + Voyage embeddings

## Environment variables

| Variable | Purpose |
|----------|---------|
| `SUPABASE_URL_DEV` | Supabase project URL (dev); wins over `SUPABASE_URL` when set |
| `SUPABASE_URL` | Used when `SUPABASE_URL_DEV` is unset |
| `SUPABASE_SERVICE_ROLE_KEY` | **Preferred** API key when set (wins over `SUPABASE_KEY` if both are set) |
| `SUPABASE_KEY` | Fallback key; use **service role** for `UPDATE` on `tracks_ai` (publishable keys often cannot write) |
| `VOYAGE_API_KEY` | [Voyage API key](https://docs.voyageai.com/docs/api-key-and-installation) |
| `VOYAGE_MODEL` | Optional; default `voyage-3.5-lite` |
| `VOYAGE_INPUT_TYPE` | Optional; default `document` (use `query` for search queries only) |
| `VOYAGE_OUTPUT_DIMENSION` | Optional; set to match your pgvector column width if the model supports it (e.g. matryoshka). If omitted, the model’s default dimension is used. |
| `VOYAGE_VECTOR_COLUMN` | Optional; default `voyage_ai_embedding`. Voyage vectors are stored here; legacy OpenAI vectors stay in `embedding`. Rows are selected where this column `IS NULL` until filled. |
| `VOYAGE_MAX_RETRIES` | Optional; default `3` |
| `VOYAGE_TIMEOUT` | Optional; seconds |
| `DATABASE_URL` | Optional. **Postgres connection URI** (Supabase: **Settings → Database**). If set, embedding **writes** use direct SQL and **bypass PostgREST** (fixes stubborn `UPDATE … affected 0 rows` when the HTTP API does not apply updates). Same names supported: `SUPABASE_DATABASE_URL`, `POSTGRES_URL`. Requires `psycopg` (`pip install 'psycopg[binary]'`). Prefer the **Connection pooling** URI (pooler host, port **6543**) if **direct** (port **5432**) fails with IPv6 / “No route to host” on your network. |

Place values in **`.env`** at the repo root. Scripts call `load_dotenv()` for that file so you do not need to `export` manually.

**Debug credentials:** from the repo root, run `python embed_pipeline.py check-env`. It prints which URL and key variables win, the JWT `role` (should be `service_role` for batch writes), and the project `ref` — without printing secrets.

### Troubleshooting: “Processed N” but nothing in the database

- **RLS:** Batch scripts need an API key that is allowed to `UPDATE` these tables. Prefer **`SUPABASE_SERVICE_ROLE_KEY`** (or `SUPABASE_KEY` set to the service role secret). Publishable/anon keys often can read but not write.
- **Wrong project:** Confirm `SUPABASE_URL_DEV` matches the project you have open in the dashboard.
- **`UPDATE … affected 0 rows` (PostgREST):** If `check-env` shows `service_role` but PATCH still updates zero rows, set **`DATABASE_URL`** to the **database password** connection string from the dashboard and rerun — writes will use Postgres directly while reads still use the Supabase client.
- After a failed write, the embedder **raises** if PostgREST returns zero updated rows (instead of only incrementing a counter).

### Billing and rate limits

Without a payment method on the Voyage dashboard, accounts are limited to very low RPM. Add a payment method at [dashboard.voyageai.com](https://dashboard.voyageai.com/) for standard limits, or use **`--sleep-seconds`** (e.g. `20`) between batches when testing.

### Parallel Voyage requests

Use **`--embed-workers N`** (default `1`) to run up to `N` concurrent **chunk** embed calls in a thread pool. Each iteration loads **`batch-size × embed-workers`** rows so there are enough texts to split into multiple API chunks. Higher `N` increases throughput until you hit **rate limits** (429); reduce `N` or add **`--sleep-seconds`** if that happens.

## Verify pgvector column width

Before a full backfill, confirm the `vector(N)` dimension in Supabase matches your Voyage output (or set `VOYAGE_OUTPUT_DIMENSION` if supported for your model):

```sql
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'tracks_ai'
  AND column_name IN ('embedding', 'voyage_ai_embedding');
```

**Important:** The embed scripts fill **`voyage_ai_embedding`** (or `VOYAGE_VECTOR_COLUMN`), not `embedding`. If almost every row already has OpenAI in `embedding`, a filter on `embedding IS NULL` would only return the rare row still missing OpenAI—use the Voyage column for backfill progress instead.

If the stored dimension does not match, alter the column or pick a different `VOYAGE_MODEL` / `VOYAGE_OUTPUT_DIMENSION` per [Voyage embeddings docs](https://docs.voyageai.com/docs/embeddings).

## Workflow

### 1. EDA (read-only)

```bash
cd /path/to/bon-voyager
python embed_pipeline.py eda --table tracks_ai --sample 20
python embed_pipeline.py eda --table playlists_ai --sample 20
```

Or run `python sbase_embeddings_rerun.py` with the same flags.

### 2. Smoke test (few rows)

```bash
python embed_pipeline.py embed-tracks --limit 5 --dry-run
python embed_pipeline.py embed-tracks --limit 5
```

### 3. Full Voyage backfill (parallel to OpenAI `embedding`)

Embeds rows where **`voyage_ai_embedding` is null**; OpenAI **`embedding`** is unchanged.

To **recompute** Voyage vectors for everyone (including rows that already have `voyage_ai_embedding`), use **`--replace-all`** (optionally with `--limit` for tests).

```bash
python embed_pipeline.py embed-tracks --replace-all --sleep-seconds 0
```

### 4. All tables (tracks, then artists, then playlists)

```bash
python embed_pipeline.py embed-all --limit 10
```

### 5. Track universe plot (2D t-SNE)

Full detail: [VISUALIZATION.md](VISUALIZATION.md) (outputs, caching, `viz` subcommand).

Requires `scikit-learn`, `matplotlib`, and for **interactive** output **`plotly`**:

```bash
pip install scikit-learn matplotlib plotly
```

**Static PNG** (colored by KMeans clusters + keyword hints printed to the terminal):

```bash
python sbase_embeddings_rerun.py --table tracks_ai --sample 1500 \
  --plot-universe tracks_universe.png --embedding-column voyage_ai_3p5_embed
```

**Interactive HTML** (zoom/pan, hover shows title/artist/tags/id; open the file in a browser):

```bash
python embed_pipeline.py eda --sample 1500 \
  --plot-universe tracks_universe.html --embedding-column voyage_ai_3p5_embed
```

**D3 HTML** (zoom/pan, text filter, sidebar cluster keywords; no extra Python deps beyond sklearn):

```bash
python embed_pipeline.py eda --sample 1500 \
  --plot-universe tracks_universe.html \
  --d3-universe tracks_universe_d3.html \
  --embedding-column voyage_ai_3p5_embed
```

**Cache the sample** so you can re-run plots without pulling rows from Supabase each time. The JSON file is written on first use (or when you **`--refresh-cache`**):

```bash
mkdir -p viz
python embed_pipeline.py eda --sample 1500 \
  --sample-cache viz/tracks_ai_sample_1500.json \
  --plot-universe tracks_universe.html \
  --d3-universe tracks_universe_d3.html \
  --embedding-column voyage_ai_3p5_embed
# Later (offline from Supabase for the sample step):
python embed_pipeline.py viz --from-cache viz/tracks_ai_sample_1500.json \
  --d3 tracks_universe_d3.html --plotly tracks_universe.html
```

Optional: **`--plot-clusters K`** to fix the number of KMeans clusters (default is automatic). Use a smaller **`--sample`** if PostgREST hits a statement timeout.

**Semantic search (Voyage query vs cached vectors)** — for analysis only, e.g. `python embed_pipeline.py search --from-cache viz/sample.json --embedding-column voyage_ai_3p5_embed "chill vibes"`, or `viz ... --search "..."` with Plotly/D3. **Live search in the browser** (no API key in JS): `python embed_pipeline.py cosmos --cache viz/tracks_sample_1500.json --embedding-column …` then open http://127.0.0.1:8765/ . See [VISUALIZATION.md](VISUALIZATION.md#semantic-search-natural-language-vs-cached-vectors).

### 6. Backfill validation

- Re-run EDA or spot-check rows in Supabase.  
- Run nearest-neighbor queries in SQL or your app.  
- If similarity quality is wrong, confirm `combined_tags` text and `VOYAGE_MODEL` / dimensions.

## music-tagging scripts

Legacy CLIs under `music-tagging/` load the parent `.env` and call the same `voyage_embed` modules:

```bash
python music-tagging/generate_embeddings.py --limit 10
python music-tagging/generate_artist_embeddings.py --limit 10
python music-tagging/generate_playlist_embeddings.py --limit 10
```

See also [music-tagging/EMBEDDINGS.md](music-tagging/EMBEDDINGS.md) for the original catalog/tag pipeline.
