"""
Voyage-embed tracks_ai rows (combined_tags -> embedding).

Prefer from repo root: python embed_pipeline.py embed-tracks

Env: SUPABASE_URL_DEV or SUPABASE_URL; SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY (service role wins if both keys set). VOYAGE_API_KEY.
Optional: VOYAGE_MODEL (default voyage-3.5-lite), VOYAGE_OUTPUT_DIMENSION (must match pgvector column).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

from voyage_embed.env import get_supabase_client
from voyage_embed.tracks import run_embed_tracks


def main() -> None:
    p = argparse.ArgumentParser(description="Voyage embeddings for tracks_ai.")
    p.add_argument("--replace-all", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--embed-workers", type=int, default=1, help="Concurrent Voyage chunk embeds (default 1).")
    args = p.parse_args()
    lim = args.limit if args.limit and args.limit > 0 else None
    run_embed_tracks(
        get_supabase_client(),
        replace_all=args.replace_all,
        limit=lim,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        embed_workers=max(1, args.embed_workers),
    )


if __name__ == "__main__":
    main()
