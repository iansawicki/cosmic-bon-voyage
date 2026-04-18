"""Helpers for Supabase updates that must persist a vector column."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from supabase import Client

from voyage_embed.env import get_database_url


def float_vector(emb: list[float] | list[Any]) -> list[float]:
    return [float(x) for x in emb]


def _pg_vector_literal(emb: list[float] | list[Any]) -> str:
    return "[" + ",".join(str(float(x)) for x in emb) + "]"


@contextmanager
def optional_pg_connection() -> Iterator[Any]:
    """
    If ``DATABASE_URL`` (or ``SUPABASE_DATABASE_URL`` / ``POSTGRES_URL``) is set,
    yield a psycopg connection for direct SQL writes; otherwise yield ``None``.
    """
    url = get_database_url()
    if not url:
        yield None
        return
    try:
        import psycopg
    except ImportError as e:
        raise RuntimeError(
            "A database URL is set but psycopg is not installed. "
            "Install with: pip install 'psycopg[binary]'"
        ) from e
    try:
        conn = psycopg.connect(url, connect_timeout=60)
    except Exception as e:
        msg = str(e).lower()
        hint = ""
        if "no route to host" in msg or "2600:" in str(e) or "ipv6" in msg:
            hint = (
                "\n\nThis often happens when the hostname resolves to IPv6 first and your network "
                "cannot reach it. In Supabase: Settings → Database → copy the "
                "**Connection pooling** URI (pooler host, usually port **6543**), not only the "
                "**Direct connection** URI (port 5432). Session pooler is a good default for scripts."
            )
        elif "connection refused" in msg or "timed out" in msg:
            hint = (
                "\n\nCheck firewall/VPN, that the password in the URI is correct, and try the "
                "pooler URI (port 6543) from Supabase if direct (5432) fails."
            )
        raise RuntimeError(
            f"DATABASE_URL is set but Postgres connection failed: {e}.{hint}"
        ) from e
    try:
        yield conn
    finally:
        conn.close()


def _update_one_pg(
    conn: Any,
    *,
    table: str,
    payload: dict[str, Any],
    pk_column: str,
    pk_value: Any,
    verify_column: str,
    context: str,
) -> None:
    from psycopg import sql

    extra = f" {context}" if context else ""
    set_fragments = []
    params: list[Any] = []
    for col, val in payload.items():
        ident = sql.Identifier(col)
        if isinstance(val, list):
            set_fragments.append(sql.SQL("{} = %s::vector").format(ident))
            params.append(_pg_vector_literal(val))
        else:
            set_fragments.append(sql.SQL("{} = %s").format(ident))
            params.append(val)

    stmt = sql.SQL("UPDATE {} SET {} WHERE {} = %s RETURNING {}").format(
        sql.Identifier(table),
        sql.SQL(", ").join(set_fragments),
        sql.Identifier(pk_column),
        sql.Identifier(verify_column),
    )
    params.append(pk_value)

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(stmt, params)
            row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"UPDATE {table} affected 0 rows for {pk_column}={pk_value!r}.{extra} "
            "No matching row, or connection points at a different database than the API."
        )
    if row[0] is None:
        raise RuntimeError(
            f"After UPDATE, {verify_column!r} is still null for {pk_column}={pk_value!r}.{extra} "
            "Check vector(N) dimension vs embedding length and column type."
        )


def update_one_or_raise(
    supabase: Client,
    *,
    table: str,
    payload: dict[str, Any],
    pk_column: str,
    pk_value: Any,
    verify_column: str,
    context: str = "",
    pg_conn: Any | None = None,
) -> None:
    """
    Write one row. If ``pg_conn`` is set, uses direct Postgres (bypasses PostgREST/RLS).
    Otherwise uses Supabase PATCH + verify SELECT.
    """
    extra = f" {context}" if context else ""

    if pg_conn is not None:
        _update_one_pg(
            pg_conn,
            table=table,
            payload=payload,
            pk_column=pk_column,
            pk_value=pk_value,
            verify_column=verify_column,
            context=context,
        )
        return

    try:
        patch_resp = (
            supabase.table(table).update(payload).eq(pk_column, pk_value).execute()
        )
    except Exception as e:
        raise RuntimeError(
            f"UPDATE {table} failed for {pk_column}={pk_value!r}.{extra} {e}"
        ) from e

    patched = getattr(patch_resp, "data", None) or []
    if not patched:
        raise RuntimeError(
            f"UPDATE {table} affected 0 rows for {pk_column}={pk_value!r}.{extra} "
            "PostgREST often returns 200 with an empty body when RLS blocks UPDATE or no row matches. "
            "Set DATABASE_URL to the Postgres connection string (Settings → Database) to write via SQL "
            "and bypass PostgREST, or use the service_role JWT (python embed_pipeline.py check-env)."
        )

    sel = (
        supabase.table(table)
        .select(verify_column)
        .eq(pk_column, pk_value)
        .limit(1)
        .execute()
    )
    rows = getattr(sel, "data", None) or []
    if not rows:
        raise RuntimeError(
            f"After UPDATE, SELECT returned no row for {pk_column}={pk_value!r}.{extra} "
            "Confirm RLS allows SELECT and UPDATE for this key (use service_role for batch jobs), "
            "and that .env points at the same Supabase project as the dashboard."
        )
    val = rows[0].get(verify_column)
    if val is None:
        raise RuntimeError(
            f"After UPDATE, {verify_column!r} is still null for {pk_column}={pk_value!r}.{extra} "
            "RLS often allows SELECT but blocks UPDATE—set SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) to the service_role JWT. "
            "Also confirm vector(N) dimension matches the model output (e.g. 1024)."
        )
