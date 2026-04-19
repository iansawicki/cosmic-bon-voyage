"""CLI: python -m voyage_embed.track_tagging"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
load_dotenv()


def main() -> None:
    import os

    p = argparse.ArgumentParser(description="Tag tracks_ai via Vertex Gemini (structured JSON).")
    p.add_argument("--replace-all", action="store_true", help="Tag all rows, not only tagging_status=pending")
    p.add_argument("--limit", type=int, default=None, metavar="N")
    p.add_argument("--dry-run", action="store_true", help="No API calls or DB writes")
    p.add_argument("--workers", type=int, default=1, metavar="N", help="Concurrent taggers (default 1)")
    p.add_argument("--model", type=str, default=None, help="Override GEMINI_TAG_MODEL")
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        metavar="NAME",
        help="Prompt module under prompts/ (no .py), e.g. prompt_1. Overrides TAG_PROMPT env.",
    )
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    if args.prompt:
        os.environ["TAG_PROMPT"] = args.prompt.strip().removesuffix(".py")

    from voyage_embed.env import get_supabase_client
    from voyage_embed.track_tagging.run import run_tag_tracks

    sb = get_supabase_client()
    out = run_tag_tracks(
        sb,
        replace_all=args.replace_all,
        limit=args.limit,
        dry_run=args.dry_run,
        workers=max(1, args.workers),
        model=args.model,
        temperature=args.temperature,
    )
    print(out)


if __name__ == "__main__":
    main()
