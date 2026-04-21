import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { getSignedUrl } from "npm:@aws-sdk/cloudfront-signer";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

function encodeObjectKeyAsUrlPath(key: string): string {
  return key.split("/").map((segment) => encodeURIComponent(segment)).join("/");
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  try {
    const { assetName, expiresInSeconds = 3600 } = await req.json();
    if (!assetName || typeof assetName !== "string") {
      return new Response(
        JSON.stringify({ error: "assetName is required" }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const domain = Deno.env.get("CLOUDFRONT_DOMAIN");
    const keyPairId = Deno.env.get("CLOUDFRONT_KEY_PAIR_ID");
    const privateKey = Deno.env.get("CLOUDFRONT_PRIVATE_KEY");
    const pathPrefix = Deno.env.get("CLOUDFRONT_PATH_PREFIX") ?? "";

    if (!domain || !keyPairId || !privateKey) {
      return new Response(
        JSON.stringify({ error: "Missing CloudFront configuration" }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const decodedPrivateKey = base64ToString(privateKey);

    const path = pathPrefix ? `${pathPrefix.replace(/\/$/, "")}/${assetName}` : assetName;
    const baseUrl = `https://${domain}/${encodeObjectKeyAsUrlPath(path)}`;
    const dateLessThan = new Date(Date.now() + expiresInSeconds * 1000);

    const signedUrl = getSignedUrl({
      url: baseUrl,
      keyPairId,
      privateKey: decodedPrivateKey,
      dateLessThan,
    });

    return new Response(
      JSON.stringify({ signedUrl }),
      { headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  } catch (e) {
    return new Response(
      JSON.stringify({ error: String(e) }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});

// Decode base64 → string (browser-safe)
function base64ToString(b64) {
  return new TextDecoder().decode(Uint8Array.from(atob(b64), c => c.charCodeAt(0)));
}