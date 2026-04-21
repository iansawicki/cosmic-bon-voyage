import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "jsr:@supabase/supabase-js@2";
import OpenAI from "npm:openai@^4";

// NLP station search.
//
// Secrets required (Dashboard → Edge Functions → Secrets):
//   - OPENAI_API_KEY
//   - SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (auto-injected by Supabase)
//
// Contract used by the Flutter client (lib/services/supabase_service.dart → search):
//   POST /functions/v1/search-music
//   body:  { "query": string }
//   200:   Array<{ result_type: "curated_station" | "ai_station", result_id: string, ... }>
//   4xx:   { error: string, code: string }
//   5xx:   { error: string, code: string }

const app = new Hono();

app.use("*", logger(console.log));

app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization", "apikey"],
    allowMethods: ["POST", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

const EMBEDDING_MODEL = "text-embedding-3-small";
const MAX_QUERY_LENGTH = 500;
const MIN_QUERY_LENGTH = 3;

function getSupabaseClient() {
  return createClient(
    Deno.env.get("SUPABASE_URL") ?? "",
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "",
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false,
      },
    },
  );
}

app.post("/*", async (c) => {
  let payload: unknown;
  try {
    payload = await c.req.json();
  } catch (_) {
    return c.json(
      { error: "Request body must be valid JSON.", code: "invalid_body" },
      400,
    );
  }

  const query = (payload as { query?: unknown })?.query;
  if (typeof query !== "string") {
    return c.json(
      { error: "Missing 'query' string.", code: "missing_query" },
      400,
    );
  }

  const trimmed = query.trim();
  if (trimmed.length < MIN_QUERY_LENGTH) {
    return c.json(
      {
        error: `Query must be at least ${MIN_QUERY_LENGTH} characters.`,
        code: "query_too_short",
      },
      400,
    );
  }
  if (trimmed.length > MAX_QUERY_LENGTH) {
    return c.json(
      {
        error: `Query must be at most ${MAX_QUERY_LENGTH} characters.`,
        code: "query_too_long",
      },
      400,
    );
  }

  const openAiKey = Deno.env.get("OPENAI_API_KEY");
  if (!openAiKey || openAiKey.trim() === "") {
    console.error("OPENAI_API_KEY is not set");
    return c.json(
      { error: "Search is not configured.", code: "missing_openai_key" },
      503,
    );
  }

  // OpenAI embedding — 401 here means the configured key was rejected
  // (rotated / revoked / wrong project). Surface as 502 upstream_auth so
  // ops can distinguish from our own auth failures.
  let embedding: number[];
  try {
    const openai = new OpenAI({ apiKey: openAiKey });
    const res = await openai.embeddings.create({
      model: EMBEDDING_MODEL,
      input: trimmed,
    });
    const vector = res.data?.[0]?.embedding;
    if (!Array.isArray(vector) || vector.length === 0) {
      console.error("openai embeddings.create returned empty vector", res);
      return c.json(
        { error: "Embedding response was empty.", code: "empty_embedding" },
        502,
      );
    }
    embedding = vector;
  } catch (err) {
    const status = (err as { status?: number })?.status;
    const message = (err as { message?: string })?.message ?? "Unknown error";
    console.error("openai embeddings.create failed", { status, message });
    if (status === 401) {
      return c.json(
        {
          error: "Embedding provider rejected the API key.",
          code: "upstream_auth",
        },
        502,
      );
    }
    if (status === 429) {
      return c.json(
        { error: "Embedding provider is rate-limiting.", code: "upstream_429" },
        503,
      );
    }
    return c.json(
      { error: "Embedding request failed.", code: "upstream_error" },
      502,
    );
  }

  try {
    const supabase = getSupabaseClient();
    const { data, error } = await supabase.rpc("search_music", {
      query_embedding: embedding,
    });
    if (error) {
      console.error("rpc search_music failed", error);
      return c.json(
        { error: "Search lookup failed.", code: "rpc_error" },
        500,
      );
    }
    return c.json(data ?? []);
  } catch (err) {
    console.error("search_music: unexpected", err);
    return c.json(
      { error: "Internal server error.", code: "internal_error" },
      500,
    );
  }
});

Deno.serve(app.fetch);
