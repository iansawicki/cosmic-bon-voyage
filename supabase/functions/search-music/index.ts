import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "jsr:@supabase/supabase-js@2";
import { embedSearchQuery, VoyageApiError } from "./voyage_query_embed.ts";

// NLP station search.
//
// Secrets required (Dashboard → Edge Functions → Secrets):
//   - VOYAGE_API_KEY (required)
//   - Optional: VOYAGE_MODEL (default voyage-3.5-lite), VOYAGE_OUTPUT_DIMENSION,
//     VOYAGE_INPUT_TYPE (default query for user search strings — use document only if you know you need it)
//   - SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (auto-injected by Supabase)
//
// Contract used by the Flutter client (lib/services/supabase_service.dart → search):
//   POST /functions/v1/search-music
//   body:  { "query": string }
//   200:   Array<{ result_type: "curated_station" | "ai_station", result_id: string, ... }>
//   DB:    rpc("search_music_voyage", { query_embedding }) — see supabase/sql_functions_local/search_music_voyage.sql
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

  const voyageKey = Deno.env.get("VOYAGE_API_KEY")?.trim();
  if (!voyageKey) {
    console.error("VOYAGE_API_KEY is not set");
    return c.json(
      { error: "Search is not configured.", code: "missing_voyage_key" },
      503,
    );
  }

  // Voyage embedding — 401 means the configured key was rejected.
  let embedding: number[];
  try {
    embedding = await embedSearchQuery(trimmed);
  } catch (err) {
    if (err instanceof VoyageApiError) {
      if (err.voyageCode === "missing_voyage_key") {
        return c.json(
          { error: "Search is not configured.", code: "missing_voyage_key" },
          503,
        );
      }
      if (err.voyageCode === "empty_embedding") {
        console.error("voyage returned empty embedding", err.detail);
        return c.json(
          { error: "Embedding response was empty.", code: "empty_embedding" },
          502,
        );
      }
      const status = err.status;
      console.error("voyage embeddings failed", {
        status,
        detail: err.detail,
        message: err.message,
      });
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
          {
            error: "Embedding provider is rate-limiting.",
            code: "upstream_429",
          },
          503,
        );
      }
      return c.json(
        { error: "Embedding request failed.", code: "upstream_error" },
        502,
      );
    }
    const message = (err as { message?: string })?.message ?? "Unknown error";
    console.error("embedSearchQuery unexpected", message, err);
    return c.json(
      { error: "Embedding request failed.", code: "upstream_error" },
      502,
    );
  }

  try {
    const supabase = getSupabaseClient();
    const { data, error } = await supabase.rpc("search_music_voyage", {
      query_embedding: embedding,
    });
    if (error) {
      console.error("rpc search_music_voyage failed", error);
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
