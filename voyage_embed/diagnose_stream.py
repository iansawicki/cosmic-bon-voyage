"""
Standalone diagnostic: asks the Supabase batch signer for a signed URL, then HEADs
CloudFront to see whether the signature actually works — independent of the D3 page.

Usage:
  python -m voyage_embed.diagnose_stream --asset "catalog.app/Everything I Need.mp3"

Reads the same env vars as ``cosmos_server`` (SUPABASE_URL[_DEV] + a JWT). Prints:
  - signed URL host/path (no signature bytes)
  - HEAD status + a short body/XML snippet on non-2xx (so you can see whether CloudFront
    reports a signature mismatch, missing key, or access denial)
"""

from __future__ import annotations

import argparse
import http.client
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

from voyage_embed.cosmic_stream import (  # noqa: E402  (env must load first)
    batch_sign_asset_urls_http,
    encode_audio_key_for_signer,
    normalize_signed_url_for_http,
)


def _head(url: str) -> tuple[int, dict[str, str], bytes]:
    """HEAD ``url`` and return (status, headers, body). Captures HTTPError body too."""
    req = urllib.request.Request(url, method="HEAD")
    req.add_header("User-Agent", "bon-voyager-diag/1.0")
    req.add_header("Accept", "*/*")
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            return resp.status, dict(resp.headers.items()), b""
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read()
        except OSError:
            pass
        return e.code, dict(e.headers.items()), body


def _get_small(url: str, n: int = 4096) -> tuple[int, dict[str, str], bytes]:
    """Range GET first few KB. Some CDNs only reveal detailed errors on GET, not HEAD."""
    req = urllib.request.Request(url, method="GET")
    req.add_header("User-Agent", "bon-voyager-diag/1.0")
    req.add_header("Accept", "*/*")
    req.add_header("Range", f"bytes=0-{n - 1}")
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = resp.read(n)
            return resp.status, dict(resp.headers.items()), body
    except urllib.error.HTTPError as e:
        body = b""
        try:
            body = e.read()
        except OSError:
            pass
        return e.code, dict(e.headers.items()), body


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--asset",
        required=True,
        help="Asset name as stored in audio_key (e.g. 'catalog.app/Song Name.mp3')",
    )
    args = ap.parse_args()

    asset = args.asset
    signer_key = encode_audio_key_for_signer(asset)
    print(f"audio_key:  {asset!r}")
    print(f"signer_key: {signer_key!r}  (stripped catalog.app/ + URL-encoded)")

    try:
        out = batch_sign_asset_urls_http([signer_key])
    except RuntimeError as e:
        print(f"signer error: {e}", file=sys.stderr)
        return 2
    url = out.get(signer_key)
    if not isinstance(url, str) or not url:
        print(f"signer did not return a URL; payload keys: {list(out)}", file=sys.stderr)
        return 2

    parts = urlsplit(url)
    print(f"signed host: {parts.netloc}")
    print(f"signed path (raw): {parts.path}")
    print(f"signed path contains space? {' ' in parts.path}")
    print(f"signed path contains %20? {'%20' in parts.path}")

    normalized = normalize_signed_url_for_http(url)
    norm_parts = urlsplit(normalized)
    print(f"normalized path: {norm_parts.path}")

    for label, u in (("raw", url), ("normalized", normalized)):
        print(f"\n-- HEAD ({label}) --")
        try:
            status, headers, body = _head(u)
        except http.client.InvalidURL as e:
            print(f"InvalidURL: {e}")
            continue
        print(f"status: {status}")
        interesting = {k: v for k, v in headers.items() if k.lower() in {
            "content-type", "content-length", "x-amz-cf-id", "x-cache",
            "x-amz-cf-pop", "server", "x-amz-error-code", "x-amz-error-message",
        }}
        for k, v in interesting.items():
            print(f"  {k}: {v}")
        if status >= 400 and body:
            print(f"  body: {body[:800]!r}")
        if status >= 400:
            print(f"-- GET first 4KB ({label}) --")
            try:
                gs, gh, gb = _get_small(u)
            except http.client.InvalidURL:
                continue
            print(f"  status: {gs}")
            if gb:
                print(f"  body: {gb[:800]!r}")
        if u == url and status < 400:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
