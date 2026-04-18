"""Load Supabase and Voyage settings from the environment."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any

from supabase import Client, create_client
from supabase.client import ClientOptions


def _env_nonempty(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None:
        return None
    s = v.strip()
    return s if s else None


def get_database_url() -> str | None:
    """
    Postgres connection URI for direct SQL writes (bypasses PostgREST).

    Set one of: ``DATABASE_URL``, ``SUPABASE_DATABASE_URL``, ``POSTGRES_URL``.
    In Supabase: Settings → Database → connection string (use pooler or direct; ``sslmode=require``).
    """
    return (
        _env_nonempty("DATABASE_URL")
        or _env_nonempty("SUPABASE_DATABASE_URL")
        or _env_nonempty("POSTGRES_URL")
    )


def decode_supabase_jwt_payload(secret: str) -> dict[str, Any] | None:
    """Decode JWT payload without verifying signature (local diagnostics only)."""
    parts = secret.split(".")
    if len(parts) != 3:
        return None
    payload_b64 = parts[1]
    pad = (-len(payload_b64)) % 4
    if pad:
        payload_b64 += "=" * pad
    try:
        raw = base64.urlsafe_b64decode(payload_b64.encode("ascii"))
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None


@dataclass(frozen=True)
class ResolvedSupabaseCredentials:
    """What ``get_supabase_client()`` will use after applying precedence rules."""

    url: str
    key: str
    url_source: str
    key_source: str
    jwt_role: str | None
    jwt_ref: str | None
    jwt_exp: int | None
    jwt_decode_error: str | None


def resolve_supabase_credentials() -> ResolvedSupabaseCredentials | None:
    """
    Same URL/key rules as ``get_supabase_client``:

    - URL: ``SUPABASE_URL_DEV`` if set and non-empty, else ``SUPABASE_URL``.
    - Key: ``SUPABASE_SERVICE_ROLE_KEY`` if set and non-empty, else ``SUPABASE_KEY``.

    Service role is preferred when both keys are set so a leftover publishable
    ``SUPABASE_KEY`` does not override the service role secret.
    """
    url = _env_nonempty("SUPABASE_URL_DEV") or _env_nonempty("SUPABASE_URL")
    url_src = (
        "SUPABASE_URL_DEV"
        if _env_nonempty("SUPABASE_URL_DEV")
        else ("SUPABASE_URL" if _env_nonempty("SUPABASE_URL") else "")
    )
    key = _env_nonempty("SUPABASE_SERVICE_ROLE_KEY") or _env_nonempty("SUPABASE_KEY")
    key_src = (
        "SUPABASE_SERVICE_ROLE_KEY"
        if _env_nonempty("SUPABASE_SERVICE_ROLE_KEY")
        else ("SUPABASE_KEY" if _env_nonempty("SUPABASE_KEY") else "")
    )
    if not url or not key:
        return None

    payload = decode_supabase_jwt_payload(key)
    if payload is None:
        return ResolvedSupabaseCredentials(
            url=url,
            key=key,
            url_source=url_src,
            key_source=key_src,
            jwt_role=None,
            jwt_ref=None,
            jwt_exp=None,
            jwt_decode_error="Value does not look like a JWT (expected three dot-separated segments).",
        )
    role = payload.get("role")
    if not isinstance(role, str):
        role = None
    ref = payload.get("ref")
    if not isinstance(ref, str):
        ref = None
    exp = payload.get("exp")
    exp_i = int(exp) if isinstance(exp, (int, float)) else None
    return ResolvedSupabaseCredentials(
        url=url,
        key=key,
        url_source=url_src,
        key_source=key_src,
        jwt_role=role,
        jwt_ref=ref,
        jwt_exp=exp_i,
        jwt_decode_error=None,
    )


def print_supabase_env_diagnostics() -> None:
    """Print safe diagnostics for debugging URL/key/JWT role (no secrets)."""
    dev = _env_nonempty("SUPABASE_URL_DEV")
    surl = _env_nonempty("SUPABASE_URL")
    skey = _env_nonempty("SUPABASE_KEY")
    srole = _env_nonempty("SUPABASE_SERVICE_ROLE_KEY")

    print("=== Supabase env (names only; values not printed) ===")
    print(f"  DATABASE_URL / SUPABASE_DATABASE_URL / POSTGRES_URL set: {bool(get_database_url())}")
    print(f"  SUPABASE_URL_DEV set: {bool(dev)}")
    print(f"  SUPABASE_URL set:     {bool(surl)}")
    print(f"  SUPABASE_KEY set:              {bool(skey)}")
    print(f"  SUPABASE_SERVICE_ROLE_KEY set: {bool(srole)}")
    if dev and srole:
        print(
            "  Note: Both SUPABASE_URL_DEV and SUPABASE_SERVICE_ROLE_KEY are set; "
            "URL wins from SUPABASE_URL_DEV, key wins from SUPABASE_SERVICE_ROLE_KEY (preferred over SUPABASE_KEY)."
        )
    r = resolve_supabase_credentials()
    if r is None:
        print("\n  ERROR: Missing SUPABASE_URL (or SUPABASE_URL_DEV) and/or a key "
              "(SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY).")
        return
    print("\n=== Effective client config (same as embed_pipeline uses) ===")
    print(f"  URL from:  {r.url_source}")
    print(f"  Key from:  {r.key_source}")
    host = r.url.replace("https://", "").replace("http://", "").split("/")[0]
    print(f"  URL host:  {host}")
    if r.jwt_decode_error:
        print(f"  JWT:       {r.jwt_decode_error}")
    else:
        print(f"  JWT role:  {r.jwt_role!r}  (want 'service_role' for batch UPDATE / RLS bypass)")
        if r.jwt_ref:
            print(f"  JWT ref:   {r.jwt_ref}  (compare to project ref in dashboard URL)")
        if r.jwt_exp is not None:
            print(f"  JWT exp:   {r.jwt_exp}")
    if r.jwt_role not in (None, "service_role"):
        print("\n  WARNING: Role is not service_role — RLS may block UPDATEs; embedding writes can fail with 'still null'.")
    if skey and srole and r.key_source == "SUPABASE_SERVICE_ROLE_KEY":
        print("\n  OK: Both keys were set; using SUPABASE_SERVICE_ROLE_KEY (not SUPABASE_KEY).")


@dataclass(frozen=True)
class VoyageSettings:
    model: str
    input_type: str
    output_dimension: int | None  # None = model default
    max_retries: int
    timeout: float | None


def get_supabase_client() -> Client:
    """
    URL: ``SUPABASE_URL_DEV`` if set, else ``SUPABASE_URL``.

    Key: ``SUPABASE_SERVICE_ROLE_KEY`` if set, else ``SUPABASE_KEY`` (service role
    wins when both are set).
    """
    r = resolve_supabase_credentials()
    if r is None:
        raise SystemExit(
            "Set Supabase credentials: SUPABASE_URL_DEV or SUPABASE_URL, and "
            "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY (service role recommended for writes). "
            "Run: python embed_pipeline.py check-env"
        )
    url, key = r.url, r.key
    return create_client(
        url,
        key,
        options=ClientOptions(
            postgrest_client_timeout=120,
            storage_client_timeout=10,
            schema="public",
        ),
    )


def get_voyage_vector_column() -> str:
    """
    Column for Voyage vectors (parallel to legacy OpenAI `embedding`).
    Default voyage_ai_embedding. Override with VOYAGE_VECTOR_COLUMN.
    """
    return os.environ.get("VOYAGE_VECTOR_COLUMN", "voyage_ai_embedding")


def get_voyage_settings() -> VoyageSettings:
    model = os.environ.get("VOYAGE_MODEL", "voyage-3.5-lite")
    input_type = os.environ.get("VOYAGE_INPUT_TYPE", "document")
    od = os.environ.get("VOYAGE_OUTPUT_DIMENSION")
    output_dimension = int(od) if od and od.isdigit() else None
    max_retries = int(os.environ.get("VOYAGE_MAX_RETRIES", "3"))
    timeout = os.environ.get("VOYAGE_TIMEOUT")
    t = float(timeout) if timeout else None
    return VoyageSettings(
        model=model,
        input_type=input_type,
        output_dimension=output_dimension,
        max_retries=max_retries,
        timeout=t,
    )
