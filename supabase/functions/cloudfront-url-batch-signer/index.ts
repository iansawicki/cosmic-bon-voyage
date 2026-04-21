import { createClient } from "https://esm.sh/@supabase/supabase-js@2";
import { createPrivateKey, createSign, type KeyLike } from "node:crypto";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

/** Encode each path segment so the signed Resource URL is valid HTTP (spaces etc.) and matches clients that percent-encode. */
function encodeObjectKeyAsUrlPath(key: string): string {
  return key.split("/").map((segment) => encodeURIComponent(segment)).join("/");
}

function toUrlSafeBase64(bytes: Uint8Array): string {
  return btoa(String.fromCharCode(...bytes))
    .replace(/\+/g, "-")
    .replace(/=/g, "_")
    .replace(/\//g, "~");
}

function chunkBase64(input: string): string {
  return input.match(/.{1,64}/g)?.join("\n") ?? input;
}

function decodeBase64(input: string): Uint8Array | null {
  try {
    const normalized = input.replace(/\s+/g, "").replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
    const decoded = atob(padded);
    return Uint8Array.from(decoded, (c) => c.charCodeAt(0));
  } catch {
    return null;
  }
}

function buildKeyCandidates(privateKeyRaw: string): KeyLike[] {
  const raw = privateKeyRaw.trim().replace(/\r/g, "");
  const withRealNewlines = raw.includes("\\n") ? raw.replace(/\\n/g, "\n") : raw;
  const compact = withRealNewlines.replace(/\s+/g, "");

  const pemCandidates = new Set<string>();
  pemCandidates.add(withRealNewlines);

  const hasPemHeader = /-----BEGIN [A-Z ]+-----/.test(withRealNewlines);
  if (!hasPemHeader) {
    const body = chunkBase64(compact);
    pemCandidates.add(`-----BEGIN PRIVATE KEY-----\n${body}\n-----END PRIVATE KEY-----`);
    pemCandidates.add(`-----BEGIN RSA PRIVATE KEY-----\n${body}\n-----END RSA PRIVATE KEY-----`);
  }

  const keyCandidates: KeyLike[] = [...pemCandidates].filter(Boolean);

  const derBytes = decodeBase64(compact);
  if (derBytes) {
    try {
      keyCandidates.push(createPrivateKey({ key: Buffer.from(derBytes), format: "der", type: "pkcs8" }));
    } catch {
      // try next format
    }
    try {
      keyCandidates.push(createPrivateKey({ key: Buffer.from(derBytes), format: "der", type: "pkcs1" }));
    } catch {
      // try next format
    }
  }

  return keyCandidates;
}

function trySign(policy: string, key: KeyLike): Uint8Array | null {
  try {
    const signer = createSign("RSA-SHA1");
    signer.update(policy);
    signer.end();
    return new Uint8Array(signer.sign(key));
  } catch {
    return null;
  }
}

function signCloudFrontUrl(
  url: string,
  keyPairId: string,
  privateKeyPem: string,
  expiresInSeconds: number,
): string {
  const expires = Math.floor(Date.now() / 1000) + expiresInSeconds;

  const policy = JSON.stringify({
    Statement: [
      {
        Resource: url,
        Condition: { DateLessThan: { "AWS:EpochTime": expires } },
      },
    ],
  });

  const keyCandidates = buildKeyCandidates(privateKeyPem);
  let signatureBytes: Uint8Array | null = null;

  for (const key of keyCandidates) {
    signatureBytes = trySign(policy, key);
    if (signatureBytes) break;
  }

  if (!signatureBytes) {
    throw new Error("Unable to parse CLOUDFRONT_PRIVATE_KEY in PEM/DER formats");
  }

  const safeSignature = toUrlSafeBase64(signatureBytes);
  return `${url}?Expires=${expires}&Key-Pair-Id=${encodeURIComponent(keyPairId)}&Signature=${safeSignature}`;
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    let body: { assetNames?: string[] };
    try {
      body = await req.json();
    } catch {
      return new Response(
        JSON.stringify({ error: "Invalid JSON body" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const assetNames = body.assetNames;
    if (!assetNames || !Array.isArray(assetNames) || assetNames.length === 0) {
      return new Response(
        JSON.stringify({ error: "assetNames array is required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const cloudfrontDomain = Deno.env.get("CLOUDFRONT_DOMAIN");
    const keyPairId = Deno.env.get("CLOUDFRONT_KEY_PAIR_ID");
    const privateKey = Deno.env.get("CLOUDFRONT_PRIVATE_KEY")?.replace(/\\n/g, "\n")?.trim();
    const supabaseUrl = Deno.env.get("SUPABASE_URL") || Deno.env.get("SB_URL");
    const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || Deno.env.get("SB_SERVICE_ROLE_KEY");

    console.log("[Signer] Secret presence", {
      hasCloudfrontDomain: Boolean(cloudfrontDomain),
      hasKeyPairId: Boolean(keyPairId),
      hasPrivateKey: Boolean(privateKey),
      hasSupabaseUrl: Boolean(supabaseUrl),
      hasServiceRoleKey: Boolean(serviceRoleKey),
    });

    if (!cloudfrontDomain || !keyPairId || !privateKey || !supabaseUrl || !serviceRoleKey) {
      return new Response(
        JSON.stringify({ error: "Server misconfigured" }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
      );
    }

    const supabase = createClient(supabaseUrl, serviceRoleKey);

    const CACHE_TTL_HOURS = 24;
    const SIGN_TTL_SECONDS = CACHE_TTL_HOURS * 60 * 60;
    const cutoff = new Date(Date.now() - CACHE_TTL_HOURS * 60 * 60 * 1000).toISOString();

    const { data: cached, error: cacheReadError } = await supabase
      .from("signed_url_cache")
      .select("asset_name, signed_url, signed_at")
      .in("asset_name", assetNames)
      .gte("signed_at", cutoff);

    if (cacheReadError) {
      console.error("[Signer] Cache read error", cacheReadError);
    }

    const cachedMap = new Map<string, string>();
    if (cached) {
      for (const row of cached) {
        cachedMap.set(row.asset_name, row.signed_url);
      }
    }

    const result: Record<string, string> = {};
    const toSign: string[] = [];

    for (const name of assetNames) {
      if (cachedMap.has(name)) {
        result[name] = cachedMap.get(name)!;
      } else {
        toSign.push(name);
      }
    }

    if (toSign.length > 0) {
      const upsertRows: { asset_name: string; signed_url: string; signed_at: string }[] = [];

      for (const name of toSign) {
        const resourceUrl = `https://${cloudfrontDomain}/${encodeObjectKeyAsUrlPath(name)}`;
        const signedUrl = signCloudFrontUrl(resourceUrl, keyPairId, privateKey, SIGN_TTL_SECONDS);

        console.log("[Signer] Generated URL:", signedUrl);
        
        result[name] = signedUrl;
        upsertRows.push({
          asset_name: name,
          signed_url: signedUrl,
          signed_at: new Date().toISOString(),
        });
      }

      const { error: cacheWriteError } = await supabase
        .from("signed_url_cache")
        .upsert(upsertRows, { onConflict: "asset_name" });

      if (cacheWriteError) {
        console.error("[Signer] Cache write error", cacheWriteError);
      }
    }

    console.log(`[Signer] Signed ${toSign.length} new, ${assetNames.length - toSign.length} cached, total ${assetNames.length}`);

    return new Response(JSON.stringify(result), {
      status: 200,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("[Signer] Unexpected error", err);
    return new Response(
      JSON.stringify({ error: "Internal server error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } },
    );
  }
});
