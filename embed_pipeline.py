#!/usr/bin/env python3
"""
Orchestrate EDA and Voyage embedding backfills for tracks_ai, artists_ai, playlists_ai.

  python embed_pipeline.py eda --sample 20
  python embed_pipeline.py eda --sample-cache viz/tracks_sample.json --plot-universe u.html --d3-universe u_d3.html
  python embed_pipeline.py viz --from-cache viz/tracks_sample.json --d3 tracks_d3.html
  python embed_pipeline.py embed-tracks --limit 10
  python embed_pipeline.py embed-all --limit 5 --dry-run
  python embed_pipeline.py tag-tracks --limit 10
  python embed_pipeline.py embed-all --tag-tracks-first --tag-limit 5
  python embed_pipeline.py check-env
  python embed_pipeline.py search --from-cache viz/sample.json --query "chill microdose vibes"
  python embed_pipeline.py cosmos --cache viz/tracks_sample_1500.json --embedding-column voyage_ai_3p5_embed

Environment: see USAGE.md. Uses SUPABASE_URL_DEV or SUPABASE_URL; SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY (service role preferred when both keys set), VOYAGE_API_KEY. Semantic search uses VOYAGE_API_KEY and VOYAGE_MODEL / VOYAGE_OUTPUT_DIMENSION (match stored vectors).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Load repo-root .env regardless of current working directory (so DATABASE_URL etc. apply).
load_dotenv(ROOT / ".env")
load_dotenv()


def _cmd_eda(args: argparse.Namespace) -> None:
    script = ROOT / "sbase_embeddings_rerun.py"
    cmd = [
        sys.executable,
        str(script),
        "--table",
        args.table,
        "--sample",
        str(args.sample),
    ]
    if args.json_out:
        cmd.extend(["--json-out", args.json_out])
    if args.plot_universe:
        cmd.extend(["--plot-universe", args.plot_universe])
    if args.embedding_column:
        cmd.extend(["--embedding-column", args.embedding_column])
    if getattr(args, "plot_max_points", 0) and args.plot_max_points > 0:
        cmd.extend(["--plot-max-points", str(args.plot_max_points)])
    if getattr(args, "plot_clusters", 0) and args.plot_clusters > 0:
        cmd.extend(["--plot-clusters", str(args.plot_clusters)])
    if getattr(args, "sample_cache", ""):
        cmd.extend(["--sample-cache", args.sample_cache])
    if getattr(args, "refresh_cache", False):
        cmd.append("--refresh-cache")
    if getattr(args, "d3_universe", ""):
        cmd.extend(["--d3-universe", args.d3_universe])
    subprocess.check_call(cmd)


def _print_voyage_hits_from_sims(meta_rows: list, sims, top_k: int) -> None:
    import numpy as np

    order = np.argsort(-sims)
    k = min(max(1, top_k), len(order))
    print(f"\n--- Top {k} tracks (cosine similarity to Voyage query) ---\n")
    for rank in range(k):
        j = int(order[rank])
        row = meta_rows[j]
        title = str(row.get("track_title") or row.get("title") or "")[:90]
        artist = str(row.get("artist_name") or row.get("artist") or "")[:70]
        print(f"{rank + 1:3}.  {float(sims[j]):.4f}  {title} — {artist}")


def _cmd_viz(args: argparse.Namespace) -> None:
    from voyage_embed.sample_cache import load_sample_cache
    from voyage_embed.semantic_search import semantic_search_rows, similarities_for_universe_model
    from voyage_embed.universe_viz import (
        compute_universe,
        write_d3_universe,
        write_matplotlib_universe,
        write_plotly_universe,
        _hover_html_for_row,
    )
    from sbase_embeddings_rerun import _guess_embedding_columns, _print_cluster_hints_model

    data = load_sample_cache(args.from_cache, repo_root=ROOT)
    sample: list = list(data["rows"])
    if args.sample_limit and args.sample_limit > 0:
        sample = sample[: args.sample_limit]

    want_plots = bool(args.plotly_out or args.png_out or args.d3_out)
    q = getattr(args, "search", "").strip()
    want_search = bool(q)

    if not want_plots and not want_search:
        raise SystemExit("Specify at least one of --plotly, --png, --d3, or --search.")

    model = None
    hover: list = []
    max_pts = args.plot_max_points if args.plot_max_points > 0 else 3000

    if want_plots:
        model = compute_universe(
            sample,
            args.embedding_column or None,
            max_pts,
            max(0, args.plot_clusters),
            _guess_embedding_columns,
        )
        if model is None:
            raise SystemExit(
                "Could not build universe: set --embedding-column or ensure at least five parseable vectors."
            )
        _print_cluster_hints_model(model)
        hover = [_hover_html_for_row(r) for r in model.meta_rows]

    sims = None
    if want_search:
        try:
            if model is not None:
                sims = similarities_for_universe_model(model, q)
                _print_voyage_hits_from_sims(model.meta_rows, sims, args.search_top)
            else:
                _, _, _, top_hits = semantic_search_rows(
                    sample,
                    q,
                    embedding_column=args.embedding_column or None,
                    guess_embedding_columns=_guess_embedding_columns,
                    max_points=max_pts if max_pts > 0 else 10**9,
                    top_k=args.search_top,
                )
                print(f"\n--- Top {len(top_hits)} tracks (cosine similarity to Voyage query) ---\n")
                for h in top_hits:
                    print(
                        f"{h['rank']:3}.  {h['score']:.4f}  {h['track_title'][:90]} — {h['artist_name'][:70]}"
                    )
        except ValueError as e:
            raise SystemExit(str(e)) from e

    if want_plots and model is not None:
        if args.plotly_out:
            write_plotly_universe(
                args.plotly_out,
                model,
                hover,
                similarities=sims if want_search else None,
                search_query=q if want_search else "",
            )
            print(f"Wrote Plotly HTML: {args.plotly_out}")
        if args.png_out:
            write_matplotlib_universe(args.png_out, model)
            print(f"Wrote PNG: {args.png_out}")
        if args.d3_out:
            write_d3_universe(
                args.d3_out,
                model,
                search_query=q if want_search else "",
                similarities=sims if want_search else None,
            )
            print(f"Wrote D3 HTML: {args.d3_out}")


def _cmd_search(args: argparse.Namespace) -> None:
    import json

    from voyage_embed.sample_cache import load_sample_cache
    from voyage_embed.semantic_search import semantic_search_rows
    from sbase_embeddings_rerun import _guess_embedding_columns

    data = load_sample_cache(args.from_cache, repo_root=ROOT)
    sample: list = list(data["rows"])
    if args.sample_limit and args.sample_limit > 0:
        sample = sample[: args.sample_limit]

    max_pts = args.max_points if args.max_points > 0 else 10**9
    try:
        col, _scores, meta, top_hits = semantic_search_rows(
            sample,
            args.query,
            embedding_column=args.embedding_column or None,
            guess_embedding_columns=_guess_embedding_columns,
            max_points=max_pts,
            top_k=args.top_k,
        )
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print(f"Embedding column: {col!r}  ({len(meta)} vectors scored)\n")
    for h in top_hits:
        print(f"{h['rank']:3}.  {h['score']:.4f}  {h['track_title'][:90]} — {h['artist_name'][:70]}")
        tags = h.get("combined_tags") or ""
        if str(tags).strip():
            t = str(tags)[:140]
            print(f"      {t}{'…' if len(str(tags)) > 140 else ''}")
    if args.json_out:
        p = Path(args.json_out)
        p.write_text(
            json.dumps(
                {"embedding_column": col, "query": args.query, "n_scored": len(meta), "hits": top_hits},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {p}")


def _cmd_cosmos(args: argparse.Namespace) -> None:
    from voyage_embed.cosmos_server import serve

    serve(
        cache_path=args.cache,
        embedding_column=args.embedding_column or "",
        host=args.host,
        port=args.port,
        plot_max_points=args.plot_max_points if args.plot_max_points > 0 else 3000,
        plot_clusters=max(0, args.plot_clusters),
        repo_root=ROOT,
    )


def _add_embed_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--replace-all",
        action="store_true",
        help="Re-embed rows even if embedding is already set (full table scan).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N rows (after fetch).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not call Voyage or write to Supabase.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Texts per Voyage API call (default 64).",
    )
    p.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Sleep between batches (helps with low RPM accounts).",
    )
    p.add_argument(
        "--embed-workers",
        type=int,
        default=1,
        metavar="N",
        help="Thread pool size for concurrent Voyage embed calls; row window = batch-size × N (default 1 = sequential chunks).",
    )


def _cmd_check_env(_args: argparse.Namespace) -> None:
    from voyage_embed.env import print_supabase_env_diagnostics

    print_supabase_env_diagnostics()


def main() -> None:
    parser = argparse.ArgumentParser(description="Voyage embedding pipeline for Supabase AI tables.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser(
        "check-env",
        help="Print which Supabase URL/key the client will use and JWT role (no secrets).",
    )
    p_check.set_defaults(func=_cmd_check_env)

    eda = sub.add_parser("eda", help="Run Supabase EDA (wraps sbase_embeddings_rerun.py).")
    eda.add_argument("--table", default="tracks_ai", help="Table name")
    eda.add_argument("--sample", type=int, default=2000, help="Max rows to sample")
    eda.add_argument("--json-out", default="", help="Write JSON report")
    eda.add_argument(
        "--plot-universe",
        default="",
        help="Output path: .html = interactive Plotly; .png = static matplotlib",
    )
    eda.add_argument("--embedding-column", default="", help="Vector column for plot")
    eda.add_argument("--plot-max-points", type=int, default=0, help="Cap t-SNE points (0 = script default)")
    eda.add_argument(
        "--plot-clusters",
        type=int,
        default=0,
        metavar="K",
        help="KMeans cluster count for colors (0 = auto)",
    )
    eda.add_argument(
        "--sample-cache",
        default="",
        metavar="PATH",
        help="Load/save sampled rows as JSON to skip repeated Supabase fetches",
    )
    eda.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refetch sample and overwrite --sample-cache",
    )
    eda.add_argument(
        "--d3-universe",
        default="",
        metavar="PATH",
        help="Write D3 interactive HTML in addition to --plot-universe",
    )
    eda.set_defaults(func=_cmd_eda)

    viz = sub.add_parser(
        "viz",
        help="Build universe HTML/PNG from a --sample-cache file (no Supabase fetch).",
    )
    viz.add_argument(
        "--from-cache",
        required=True,
        metavar="PATH",
        help="JSON from EDA --sample-cache",
    )
    viz.add_argument("--embedding-column", default="", help="Vector column (default: name heuristic)")
    viz.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        metavar="N",
        help="Use only first N rows from cache (0 = all cached rows)",
    )
    viz.add_argument("--plot-max-points", type=int, default=0, help="Cap t-SNE points (0 = 3000)")
    viz.add_argument(
        "--plot-clusters",
        type=int,
        default=0,
        metavar="K",
        help="KMeans clusters (0 = auto)",
    )
    viz.add_argument("--plotly", dest="plotly_out", default="", metavar="PATH", help="Plotly .html output")
    viz.add_argument("--png", dest="png_out", default="", metavar="PATH", help="matplotlib .png output")
    viz.add_argument("--d3", dest="d3_out", default="", metavar="PATH", help="D3 .html output")
    viz.add_argument(
        "--search",
        default="",
        metavar="TEXT",
        help="Natural-language query: Voyage query-embed, cosine vs rows; prints top hits; colors Plotly/D3 when combined",
    )
    viz.add_argument(
        "--search-top",
        type=int,
        default=15,
        metavar="N",
        help="Number of hits to print with --search (default 15)",
    )
    viz.set_defaults(func=_cmd_viz)

    p_search = sub.add_parser(
        "search",
        help="Rank cached rows by cosine similarity to a Voyage query embedding (analysis only; no Supabase).",
    )
    p_search.add_argument(
        "--from-cache",
        required=True,
        metavar="PATH",
        help="JSON from EDA --sample-cache",
    )
    p_search.add_argument("query", help="Search text (quote if it contains spaces)")
    p_search.add_argument("--embedding-column", default="", help="Vector column (default: name heuristic)")
    p_search.add_argument(
        "--sample-limit",
        type=int,
        default=0,
        metavar="N",
        help="Use only first N cached rows (0 = all)",
    )
    p_search.add_argument(
        "--max-points",
        type=int,
        default=0,
        metavar="N",
        help="Max rows to score (0 = no extra cap beyond cache)",
    )
    p_search.add_argument("--top-k", type=int, default=25, metavar="K", help="How many hits to show (default 25)")
    p_search.add_argument("--json-out", default="", metavar="PATH", help="Write hits JSON")
    p_search.set_defaults(func=_cmd_search)

    p_cosmos = sub.add_parser(
        "cosmos",
        help="Serve D3 track universe + live Voyage search in the browser (local HTTP; API key server-side only).",
    )
    p_cosmos.add_argument("--cache", required=True, metavar="PATH", help="Sample JSON from EDA --sample-cache")
    p_cosmos.add_argument("--embedding-column", default="", help="Vector column (default: heuristic)")
    p_cosmos.add_argument("--host", default="127.0.0.1")
    p_cosmos.add_argument("--port", type=int, default=8765)
    p_cosmos.add_argument("--plot-max-points", type=int, default=3000)
    p_cosmos.add_argument(
        "--plot-clusters",
        type=int,
        default=0,
        metavar="K",
        help="KMeans clusters for plot (0 = auto)",
    )
    p_cosmos.set_defaults(func=_cmd_cosmos)

    def _add_tag_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--replace-all",
            action="store_true",
            help="Tag all tracks, not only tagging_status=pending",
        )
        p.add_argument("--limit", type=int, default=None, metavar="N")
        p.add_argument("--dry-run", action="store_true", help="No Gemini calls or DB writes")
        p.add_argument("--tag-workers", type=int, default=1, metavar="N")
        p.add_argument("--tag-model", type=str, default=None, help="Override GEMINI_TAG_MODEL")
        p.add_argument(
            "--tag-prompt",
            type=str,
            default=None,
            metavar="NAME",
            help="Prompt module under voyage_embed/track_tagging/prompts/ (e.g. prompt_1). Sets TAG_PROMPT.",
        )
        p.add_argument("--temperature", type=float, default=0.2)

    def cmd_tag_tracks(a: argparse.Namespace) -> None:
        import os

        if getattr(a, "tag_prompt", None):
            os.environ["TAG_PROMPT"] = str(a.tag_prompt).strip().removesuffix(".py")

        from voyage_embed.env import get_supabase_client
        from voyage_embed.track_tagging.run import run_tag_tracks

        sb = get_supabase_client()
        lim = a.limit if a.limit and a.limit > 0 else None
        out = run_tag_tracks(
            sb,
            replace_all=a.replace_all,
            limit=lim,
            dry_run=a.dry_run,
            workers=max(1, a.tag_workers),
            model=a.tag_model,
            temperature=a.temperature,
        )
        print(out)

    def cmd_tracks(a: argparse.Namespace) -> None:
        from voyage_embed.env import get_supabase_client
        from voyage_embed.tracks import run_embed_tracks

        sb = get_supabase_client()
        lim = a.limit if a.limit and a.limit > 0 else None
        run_embed_tracks(
            sb,
            replace_all=a.replace_all,
            limit=lim,
            dry_run=a.dry_run,
            batch_size=a.batch_size,
            sleep_seconds=a.sleep_seconds,
            embed_workers=max(1, a.embed_workers),
        )

    def cmd_artists(a: argparse.Namespace) -> None:
        from voyage_embed.env import get_supabase_client
        from voyage_embed.artists import run_embed_artists

        sb = get_supabase_client()
        lim = a.limit if a.limit and a.limit > 0 else None
        run_embed_artists(
            sb,
            replace_all=a.replace_all,
            limit=lim,
            dry_run=a.dry_run,
            batch_size=a.batch_size,
            sleep_seconds=a.sleep_seconds,
            embed_workers=max(1, a.embed_workers),
        )

    def cmd_playlists(a: argparse.Namespace) -> None:
        from voyage_embed.env import get_supabase_client
        from voyage_embed.playlists import run_embed_playlists

        sb = get_supabase_client()
        lim = a.limit if a.limit and a.limit > 0 else None
        run_embed_playlists(
            sb,
            replace_all=a.replace_all,
            limit=lim,
            dry_run=a.dry_run,
            batch_size=a.batch_size,
            sleep_seconds=a.sleep_seconds,
            embed_workers=max(1, a.embed_workers),
        )

    def cmd_all(a: argparse.Namespace) -> None:
        import os

        if getattr(a, "tag_tracks_first", False):
            print("=== tag-tracks (Vertex Gemini) ===")
            tp = getattr(a, "tag_prompt", None)
            if tp:
                os.environ["TAG_PROMPT"] = str(tp).strip().removesuffix(".py")
            from voyage_embed.env import get_supabase_client
            from voyage_embed.track_tagging.run import run_tag_tracks

            sb = get_supabase_client()
            tlim = a.tag_limit if getattr(a, "tag_limit", None) and a.tag_limit > 0 else None
            print(
                run_tag_tracks(
                    sb,
                    replace_all=getattr(a, "tag_replace_all", False),
                    limit=tlim,
                    dry_run=getattr(a, "tag_dry_run", False),
                    workers=max(1, getattr(a, "tag_workers", 1)),
                    model=getattr(a, "tag_model", None),
                    temperature=float(getattr(a, "tag_temperature", 0.2)),
                )
            )
        print("=== embed-tracks ===")
        cmd_tracks(a)
        print("=== embed-artists ===")
        cmd_artists(a)
        print("=== embed-playlists ===")
        cmd_playlists(a)

    p_tracks = sub.add_parser("embed-tracks", help="Voyage-embed tracks_ai.combined_tags.")
    _add_embed_flags(p_tracks)
    p_tracks.set_defaults(func=cmd_tracks)

    p_artists = sub.add_parser("embed-artists", help="Voyage-embed artists_ai.combined_tags.")
    _add_embed_flags(p_artists)
    p_artists.set_defaults(func=cmd_artists)

    p_pl = sub.add_parser("embed-playlists", help="Voyage-embed playlists_ai.")
    _add_embed_flags(p_pl)
    p_pl.set_defaults(func=cmd_playlists)

    p_tag = sub.add_parser(
        "tag-tracks",
        help="Tag tracks_ai via Vertex Gemini (structured JSON). Requires tagging_runs migration.",
    )
    _add_tag_flags(p_tag)
    p_tag.set_defaults(func=cmd_tag_tracks)

    p_all = sub.add_parser("embed-all", help="Run embed-tracks, embed-artists, embed-playlists in order.")
    _add_embed_flags(p_all)
    p_all.add_argument(
        "--tag-tracks-first",
        action="store_true",
        help="Run Vertex tag-tracks first (uses --tag-limit / --tag-workers / --tag-dry-run / --tag-replace-all).",
    )
    p_all.add_argument("--tag-limit", type=int, default=None, metavar="N")
    p_all.add_argument("--tag-workers", type=int, default=1, metavar="N")
    p_all.add_argument("--tag-dry-run", action="store_true")
    p_all.add_argument("--tag-replace-all", action="store_true")
    p_all.add_argument("--tag-model", type=str, default=None)
    p_all.add_argument(
        "--tag-prompt",
        type=str,
        default=None,
        metavar="NAME",
        help="Same as tag-tracks --tag-prompt when using --tag-tracks-first.",
    )
    p_all.add_argument("--tag-temperature", type=float, default=0.2)
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
