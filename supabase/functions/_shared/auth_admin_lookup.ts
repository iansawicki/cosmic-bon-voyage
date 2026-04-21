/**
 * Email → auth user lookup uses the GoTrue admin HTTP API (`GET /auth/v1/admin/users?filter=…`)
 * because we need a targeted query by email; the JS client’s listUsers typing only documents
 * pagination — see https://supabase.com/docs/reference/javascript/auth-admin-listusers
 *
 * For **verification** (magic link eligibility), always hydrate with `auth.admin.getUserById`
 * so we read the same canonical User the dashboard uses (`email_confirmed_at`), not a possibly
 * minimal list payload — see https://supabase.com/docs/reference/javascript/auth-admin-getuserbyid
 */
import type { SupabaseClient, User } from "jsr:@supabase/supabase-js@2";

export async function lookupAuthUserByEmail(
  supabaseUrl: string,
  serviceRoleKey: string,
  normalizedEmail: string,
): Promise<{ user: Record<string, unknown> | null; ok: boolean }> {
  const base = supabaseUrl.replace(/\/$/, "");
  const url =
    `${base}/auth/v1/admin/users?filter=${encodeURIComponent(normalizedEmail)}`;
  const res = await fetch(url, {
    method: "GET",
    headers: {
      apikey: serviceRoleKey,
      Authorization: `Bearer ${serviceRoleKey}`,
    },
  });
  if (!res.ok) {
    const body = await res.text();
    console.error("Auth admin users lookup HTTP", res.status, body);
    return { user: null, ok: false };
  }
  const json = (await res.json()) as { users?: Array<Record<string, unknown>> };
  const users = json?.users ?? [];
  const existingUser = users.find((u) =>
    String(u?.email ?? "").trim().toLowerCase() === normalizedEmail
  );
  return { user: existingUser ?? null, ok: true };
}

/**
 * App-level verification via `user_metadata.app_email_verified`, else legacy
 * `email_confirmed_at` / `confirmed_at` (OAuth, older accounts).
 */
export function isAuthUserEmailVerified(user: User): boolean {
  const meta = user.user_metadata as Record<string, unknown> | undefined;
  const raw = meta?.app_email_verified;
  if (typeof raw === "boolean") return raw;
  if (typeof raw === "string") {
    const t = raw.trim().toLowerCase();
    if (t === "true") return true;
    if (t === "false") return false;
  }
  const emailAt = user.email_confirmed_at?.trim();
  if (emailAt) return true;
  const legacy = (user as User & { confirmed_at?: string | null }).confirmed_at
    ?.trim();
  return Boolean(legacy);
}

/**
 * Resolve email preflight: list lookup for existence, then `getUserById` for verification.
 */
export async function resolveAuthEmailPreflight(
  supabase: SupabaseClient,
  supabaseUrl: string,
  serviceRoleKey: string,
  normalizedEmail: string,
): Promise<
  | { ok: false }
  | { ok: true; exists: false }
  | { ok: true; exists: true; emailVerified: boolean }
> {
  const { user: row, ok: lookupOk } = await lookupAuthUserByEmail(
    supabaseUrl,
    serviceRoleKey,
    normalizedEmail,
  );
  if (!lookupOk) return { ok: false };
  if (!row) return { ok: true, exists: false };

  const id = row["id"];
  if (id == null || String(id).trim() === "") {
    return { ok: true, exists: false };
  }

  const { data, error } = await supabase.auth.admin.getUserById(String(id));
  if (error) {
    console.error("getUserById in email preflight:", error);
    return { ok: false };
  }
  if (!data?.user) {
    return { ok: true, exists: false };
  }

  return {
    ok: true,
    exists: true,
    emailVerified: isAuthUserEmailVerified(data.user),
  };
}
