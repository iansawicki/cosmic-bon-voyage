"""
Call Supabase Edge Functions that sign CloudFront URLs (cosmic-app deploys
``cloudfront-url-batch-signer``). Used by ``cosmos_server`` so keys stay off the static page.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

# Reuse signed URLs across Range requests (browser issues many per seek).
_signed_lock = threading.Lock()
_signed_cache: dict[str, tuple[str, float]] = {}
_SIGNED_CACHE_TTL_S = 20 * 3600  # signer uses 24h; stay below that


def _env_nonempty(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    s = v.strip()
    return s if s else None


def resolve_supabase_project_url() -> str | None:
    """Same host resolution as ``resolve_supabase_credentials`` (dev URL wins)."""
    return _env_nonempty("SUPABASE_URL_DEV") or _env_nonempty("SUPABASE_URL")


def resolve_jwt_for_edge_functions() -> str | None:
    """
    JWT for ``Authorization: Bearer`` + ``apikey`` when invoking Edge Functions.

    Prefer anon/publishable; service role also works from this server-only proxy.
    """
    return (
        _env_nonempty("SUPABASE_ANON_KEY")
        or _env_nonempty("SUPABASE_KEY")
        or _env_nonempty("SUPABASE_SERVICE_ROLE_KEY")
    )


def default_batch_signer_function_name() -> str:
    v = os.environ.get("SUPABASE_EDGE_BATCH_SIGNER")
    if v and v.strip():
        return v.strip()
    return "cloudfront-url-batch-signer"


_CATALOG_PREFIX = "catalog.app/"


def encode_audio_key_for_signer(audio_key: str) -> str:
    """
    Mirror of cosmic-app's ``Track.encodeAudioKey`` (``lib/models/track.dart``):
    strip the leading ``catalog.app/`` marker, then percent-encode the remainder
    like JS ``encodeURIComponent`` (space → ``%20``, slash → ``%2F``, …).

    The batch signer treats the incoming ``assetName`` as the **exact** object
    path on CloudFront/S3; cosmic-app and bon-voyager must agree on that string
    so they produce identical signatures and look up the same S3 key.
    """
    raw = audio_key.strip()
    if raw.startswith(_CATALOG_PREFIX):
        raw = raw[len(_CATALOG_PREFIX) :]
    return quote(raw, safe="")


def batch_sign_asset_urls_http(
    asset_names: list[str],
    *,
    function_name: str | None = None,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    """
    POST ``{assetNames}`` to the batch signer; returns the JSON body (name → signed URL).

    Raises ``RuntimeError`` on missing config, HTTP errors, or non-JSON error payloads.
    """
    url_base = resolve_supabase_project_url()
    key = resolve_jwt_for_edge_functions()
    if not url_base or not key:
        raise RuntimeError(
            "Missing SUPABASE_URL (or SUPABASE_URL_DEV) and a JWT "
            "(SUPABASE_ANON_KEY, SUPABASE_KEY, or SUPABASE_SERVICE_ROLE_KEY)"
        )
    fn = (function_name or default_batch_signer_function_name()).strip()
    edge_url = f"{url_base.rstrip('/')}/functions/v1/{fn}"
    payload = json.dumps({"assetNames": asset_names}).encode("utf-8")
    req = urllib.request.Request(
        edge_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            "apikey": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            err_json = json.loads(err_body)
            if isinstance(err_json, dict) and "error" in err_json:
                raise RuntimeError(str(err_json.get("error"))) from e
        except json.JSONDecodeError:
            pass
        raise RuntimeError(f"Edge function HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Edge request failed: {e.reason}") from e

    try:
        out = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError("Signer returned non-JSON") from e
    if not isinstance(out, dict):
        raise RuntimeError("Signer returned non-object JSON")
    return out


def get_or_cache_signed_url(audio_key: str, *, force_refresh: bool = False) -> str:
    """
    Return a signed CDN URL for the DB's ``audio_key`` (e.g.
    ``catalog.app/Song Name.mp3``), caching it so repeated Range requests do not
    call the Edge function every time.

    Cosmic-app's client transforms the key before signing (strip
    ``catalog.app/`` + ``encodeURIComponent``) so the signer — and the signed
    CloudFront ``Resource`` — match the real S3 object path. We do the same
    here; the in-memory cache keys on the original ``audio_key`` so callers
    stay ignorant of the transform.

    Pass ``force_refresh=True`` to bypass the in-memory cache (e.g. after a 403
    where a cached URL may have come from an old signer deploy).
    """
    now = time.time()
    if not force_refresh:
        with _signed_lock:
            hit = _signed_cache.get(audio_key)
            if hit and hit[1] > now:
                return hit[0]
    signer_key = encode_audio_key_for_signer(audio_key)
    out = batch_sign_asset_urls_http([signer_key])
    err = out.get("error") if isinstance(out.get("error"), str) else None
    if err:
        raise RuntimeError(err)
    url = out.get(signer_key)
    if not url or not isinstance(url, str):
        raise RuntimeError("Signer did not return a URL for this asset")
    with _signed_lock:
        _signed_cache[audio_key] = (url, now + _SIGNED_CACHE_TTL_S)
    parts = urlsplit(url)
    sys.stderr.write(
        f"[cosmic_stream] Signed audio_key={audio_key!r} "
        f"signer_key={signer_key!r} host={parts.netloc} path={parts.path}\n"
    )
    return url


def evict_signed_url(audio_key: str) -> None:
    """Drop the in-memory entry so the next call re-signs (used on upstream 403)."""
    with _signed_lock:
        _signed_cache.pop(audio_key, None)


_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


def guess_audio_content_type(asset_name: str) -> str | None:
    """
    Map file extension to a MIME type Chrome/Safari accept in ``<audio>``.

    S3/CloudFront often serve ``application/octet-stream``; without a real
    ``audio/*`` type the browser reports “no supported source was found.”
    """
    lower = asset_name.lower().rsplit("/", 1)[-1]
    dot = lower.rfind(".")
    ext = lower[dot + 1 :] if dot >= 0 else ""
    return {
        "mp3": "audio/mpeg",
        "mp2": "audio/mpeg",
        "flac": "audio/flac",
        "m4a": "audio/mp4",
        "mp4": "audio/mp4",
        "aac": "audio/aac",
        "ogg": "audio/ogg",
        "oga": "audio/ogg",
        "opus": "audio/ogg",
        "wav": "audio/wav",
        "aif": "audio/aiff",
        "aiff": "audio/aiff",
        "webm": "audio/webm",
    }.get(ext)


def normalize_signed_url_for_http(url: str) -> str:
    """
    Percent-encode the URL path so :mod:`http.client` accepts it. CloudFront
    signed URLs often include raw spaces and commas in object keys (e.g.
    ``Sea Above, Sky Below.flac``); Python raises ``InvalidURL`` for those unless
    encoded (``%20``, ``%2C``, …). Decode-then-encode avoids double-encoding.
    """
    parts = urlsplit(url)
    path = quote(unquote(parts.path), safe="/")
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def open_upstream_for_audio_proxy(
    signed_url: str,
    *,
    range_header: str | None = None,
    method: str = "GET",
    timeout_s: float = 120.0,
) -> Any:
    """
    ``GET``/``HEAD`` the signed URL. Prefer the **exact** string from the signer when
    urllib accepts it so the request matches the signed CloudFront ``Resource``. Only
    apply :func:`normalize_signed_url_for_http` when the path has raw spaces (etc.)
    and :mod:`http.client` raises ``InvalidURL``, or as a second attempt on 403 if the
    normalized URL differs.
    """
    normalized = normalize_signed_url_for_http(signed_url)
    candidates: list[str] = []
    for u in (signed_url, normalized):
        if u not in candidates:
            candidates.append(u)

    last_http: urllib.error.HTTPError | None = None
    for idx, url in enumerate(candidates):
        req = urllib.request.Request(url, method=method)
        if range_header:
            req.add_header("Range", range_header)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        req.add_header("Accept", "*/*")
        try:
            return urllib.request.urlopen(req, timeout=timeout_s)
        except http.client.InvalidURL:
            continue
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read()
            except OSError:
                pass
            if e.code == 403 and idx + 1 < len(candidates):
                sys.stderr.write(
                    "[cosmic_stream] CloudFront 403 on URL variant; retrying. "
                    f"Body: {body[:700]!r}\n"
                )
                last_http = e
                continue
            if e.code == 403:
                sys.stderr.write(
                    "[cosmic_stream] CloudFront 403 (final). Body: " f"{body[:1200]!r}\n"
                )
            raise
    if last_http is not None:
        raise last_http
    raise RuntimeError("Could not open signed URL (invalid URL for all variants)")
