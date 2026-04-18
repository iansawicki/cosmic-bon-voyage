"""
Local HTTP server: serves the D3 track universe, POST /api/search (Voyage), and
POST /api/sign-audio (JSON map from Edge) and GET /api/stream-audio?assetName=… (same-origin proxy
for the ``<audio>`` element — forwards Range to CloudFront; avoids browser CDN CORS issues).

  python -m voyage_embed.cosmos_server --cache viz/tracks_sample_1500.json --embedding-column voyage_ai_3p5_embed

Open http://127.0.0.1:8765/ — API keys stay on the server (never sent to the browser).
Set SUPABASE_URL (or SUPABASE_URL_DEV) and SUPABASE_ANON_KEY or SUPABASE_KEY for streaming.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

from voyage_embed.cosmic_stream import (
    _HOP_BY_HOP,
    batch_sign_asset_urls_http,
    evict_signed_url,
    get_or_cache_signed_url,
    guess_audio_content_type,
    open_upstream_for_audio_proxy,
)
from voyage_embed.env import get_voyage_settings
from voyage_embed.sample_cache import load_sample_cache, resolve_sample_cache_path
from voyage_embed.semantic_search import embed_query_text, stack_vectors_from_rows
from voyage_embed.universe_viz import build_d3_html_document, compute_universe


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _handler_factory(
    *,
    html_page: bytes,
    vectors_norm: np.ndarray,
    embedding_dim: int,
) -> type[BaseHTTPRequestHandler]:
    class CosmosHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

        def _send_json(self, code: int, obj: dict[str, Any]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _forward_upstream_headers(self, upstream: Any, asset: str) -> list[tuple[str, str]]:
            """Forward CDN headers; fix ``Content-Type`` when S3 sends octet-stream (Chrome ``<audio>``)."""
            pairs: list[tuple[str, str]] = []
            raw_ct: str | None = None
            for k, v in upstream.headers.items():
                kl = k.lower()
                if kl in _HOP_BY_HOP:
                    continue
                if kl == "content-type":
                    raw_ct = v
                    continue
                pairs.append((k, v))
            main = (raw_ct or "").split(";")[0].strip().lower() if raw_ct else ""
            guess = guess_audio_content_type(asset)
            if guess and (not main or main in ("application/octet-stream", "binary/octet-stream")):
                pairs.append(("Content-Type", guess))
            elif raw_ct:
                pairs.append(("Content-Type", raw_ct))
            elif guess:
                pairs.append(("Content-Type", guess))
            return pairs

        def _stream_audio_from_cdn(self, *, range_header: str | None, head_only: bool) -> None:
            """Proxy bytes from a signed CloudFront URL (same-origin for the browser)."""
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            raw = (qs.get("assetName") or [None])[0]
            if not raw:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                msg = b"Missing assetName query parameter"
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(msg)
                return
            asset = raw.strip()
            if not asset or len(asset) > 2048:
                self.send_response(400)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                msg = b"Invalid assetName"
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(msg)
                return
            force_refresh = "fresh" in qs
            try:
                signed = get_or_cache_signed_url(asset, force_refresh=force_refresh)
            except RuntimeError as e:
                err = str(e).encode("utf-8")
                self.send_response(503)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(err)
                return
            method = "HEAD" if head_only else "GET"

            def _try_open(url: str) -> Any:
                return open_upstream_for_audio_proxy(
                    url,
                    range_header=range_header if not head_only else None,
                    method=method,
                    timeout_s=120.0,
                )

            try:
                upstream = _try_open(signed)
            except urllib.error.HTTPError as first_err:
                if first_err.code == 403 and not force_refresh:
                    try:
                        first_err.read()
                    except OSError:
                        pass
                    sys.stderr.write(
                        f"[cosmos] upstream 403 for asset={asset!r}; evicting cache and re-signing once\n"
                    )
                    evict_signed_url(asset)
                    try:
                        signed = get_or_cache_signed_url(asset, force_refresh=True)
                        upstream = _try_open(signed)
                    except urllib.error.HTTPError as retry_err:
                        self.send_response(retry_err.code)
                        try:
                            hdr_pairs = self._forward_upstream_headers(retry_err, asset)
                        except Exception:  # noqa: BLE001
                            hdr_pairs = [
                                (k, v)
                                for k, v in retry_err.headers.items()
                                if k.lower() not in _HOP_BY_HOP
                            ]
                        for k, v in hdr_pairs:
                            self.send_header(k, v)
                        self.end_headers()
                        if not head_only:
                            body = retry_err.read()
                            if body:
                                self.wfile.write(body)
                        return
                    except urllib.error.URLError as retry_err:
                        self.send_response(502)
                        self.send_header("Content-Type", "text/plain; charset=utf-8")
                        msg = f"CDN request failed: {retry_err.reason!s}".encode("utf-8")
                        self.send_header("Content-Length", str(len(msg)))
                        self.end_headers()
                        if not head_only:
                            self.wfile.write(msg)
                        return
                else:
                    e = first_err
                    self.send_response(e.code)
                    try:
                        hdr_pairs = self._forward_upstream_headers(e, asset)
                    except Exception:  # noqa: BLE001
                        hdr_pairs = [
                            (k, v)
                            for k, v in e.headers.items()
                            if k.lower() not in _HOP_BY_HOP
                        ]
                    for k, v in hdr_pairs:
                        self.send_header(k, v)
                    self.end_headers()
                    if not head_only:
                        body = e.read()
                        if body:
                            self.wfile.write(body)
                    return
            except urllib.error.URLError as e:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                msg = f"CDN request failed: {e.reason!s}".encode("utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(msg)
                return
            except OSError as e:
                self.send_response(502)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                msg = str(e).encode("utf-8")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                if not head_only:
                    self.wfile.write(msg)
                return
            try:
                self.send_response(upstream.status)
                for k, v in self._forward_upstream_headers(upstream, asset):
                    self.send_header(k, v)
                self.end_headers()
                if head_only:
                    return
                while True:
                    chunk = upstream.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            finally:
                upstream.close()

        def do_OPTIONS(self) -> None:
            p = urlparse(self.path).path
            if p in ("/api/search", "/api/sign-audio", "/api/stream-audio", "/"):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "POST, GET, HEAD, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
                self.end_headers()
            else:
                self.send_error(404)

        def do_HEAD(self) -> None:
            p = urlparse(self.path).path
            if p == "/api/stream-audio":
                rng = self.headers.get("Range")
                self._stream_audio_from_cdn(range_header=rng, head_only=True)
                return
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                return
            self.send_error(404)

        def do_GET(self) -> None:
            p = urlparse(self.path).path
            if p == "/api/stream-audio":
                rng = self.headers.get("Range")
                self._stream_audio_from_cdn(range_header=rng, head_only=False)
                return
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_page)
                return
            self.send_error(404)

        def do_POST(self) -> None:
            if self.path == "/api/sign-audio":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length > 0 else b"{}"
                    body = json.loads(raw.decode("utf-8"))
                    names = body.get("assetNames")
                    if not isinstance(names, list) or not names:
                        self._send_json(400, {"error": "assetNames must be a non-empty array"})
                        return
                    str_names = [str(x).strip() for x in names if str(x).strip()]
                    if len(str_names) != len(names):
                        self._send_json(400, {"error": "assetNames must be non-empty strings"})
                        return
                    if len(str_names) > 64:
                        self._send_json(400, {"error": "Too many assets (max 64)"})
                        return
                    out = batch_sign_asset_urls_http(str_names)
                    err = out.get("error") if isinstance(out.get("error"), str) else None
                    if err:
                        self._send_json(502, {"error": err})
                        return
                    self._send_json(200, out)
                except RuntimeError as e:
                    self._send_json(503, {"error": str(e)})
                except ValueError as e:
                    self._send_json(400, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    self._send_json(500, {"error": str(e)})
                return

            if self.path != "/api/search":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8"))
                query = str(body.get("query", "")).strip()
                if not query:
                    self._send_json(400, {"error": "Missing query"})
                    return
                qvec = embed_query_text(query)
                q = np.asarray(qvec, dtype=np.float64).ravel()
                if q.shape[0] != embedding_dim:
                    self._send_json(
                        400,
                        {
                            "error": (
                                f"Query embedding dim {q.shape[0]} != stored {embedding_dim} "
                                "(check VOYAGE_MODEL / VOYAGE_OUTPUT_DIMENSION)"
                            ),
                        },
                    )
                    return
                qn = q / max(np.linalg.norm(q), 1e-12)
                sims = (vectors_norm @ qn).astype(np.float64)
                self._send_json(200, {"similarities": sims.tolist(), "query": query})
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._send_json(500, {"error": str(e)})

    return CosmosHandler


def serve(
    *,
    cache_path: str,
    embedding_column: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
    plot_max_points: int = 3000,
    plot_clusters: int = 0,
    repo_root: Path | None = None,
) -> None:
    from dotenv import load_dotenv

    from sbase_embeddings_rerun import _guess_embedding_columns

    root = repo_root or _repo_root()
    load_dotenv(root / ".env")
    load_dotenv()
    data = load_sample_cache(cache_path, repo_root=root)
    sample: list = list(data["rows"])

    model = compute_universe(
        sample,
        embedding_column or None,
        plot_max_points,
        max(0, plot_clusters),
        _guess_embedding_columns,
    )
    if model is None:
        raise SystemExit("Could not build universe: set --embedding-column or ensure >= 5 parseable vectors.")

    X, meta = stack_vectors_from_rows(model.meta_rows, model.column, len(model.meta_rows))
    if len(meta) != len(model.meta_rows) or X.shape[0] < 1:
        raise SystemExit("Could not load embedding matrix for search API.")
    vn = np.linalg.norm(X, axis=1, keepdims=True)
    vectors_norm = X / np.maximum(vn, 1e-12).astype(np.float32)

    html = build_d3_html_document(
        model,
        dynamic_search=True,
        api_search_path="/api/search",
        stream_proxy=True,
        stream_api_path="/api/sign-audio",
        stream_media_path="/api/stream-audio",
    )
    html_page = html.encode("utf-8")

    handler = _handler_factory(
        html_page=html_page,
        vectors_norm=vectors_norm.astype(np.float64),
        embedding_dim=X.shape[1],
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving cosmos at http://{host}:{port}/  (Ctrl+C to stop)")
    print(f"  cache: {resolve_sample_cache_path(cache_path, repo_root=root)}")
    print(f"  embedding column: {model.column!r}  n={model.xy.shape[0]}  dim={X.shape[1]}")
    print(f"  Voyage model: {get_voyage_settings().model!r}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    rr = _repo_root()
    load_dotenv(rr / ".env")
    load_dotenv()

    p = argparse.ArgumentParser(description="D3 track universe + live Voyage search (local server).")
    p.add_argument("--cache", required=True, help="Sample JSON from EDA --sample-cache")
    p.add_argument("--embedding-column", default="", help="Vector column (default: heuristic)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--plot-max-points", type=int, default=3000)
    p.add_argument("--plot-clusters", type=int, default=0, metavar="K")
    args = p.parse_args(argv)
    serve(
        cache_path=args.cache,
        embedding_column=args.embedding_column,
        host=args.host,
        port=args.port,
        plot_max_points=args.plot_max_points,
        plot_clusters=args.plot_clusters,
    )


if __name__ == "__main__":
    main()
