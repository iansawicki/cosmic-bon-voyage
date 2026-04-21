/**
 * Query-side Voyage embeddings for search (input_type "query").
 * Env alignment: same names as Python `voyage_embed/env.py` (VOYAGE_MODEL defaults, etc.).
 */

const VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings";

/** Default when VOYAGE_MODEL is unset — matches `get_voyage_settings()` in voyage_embed/env.py */
const DEFAULT_VOYAGE_MODEL = "voyage-3.5-lite";

export class VoyageApiError extends Error {
  readonly status: number;
  readonly detail?: string;
  /** Stable machine code for callers (e.g. `empty_embedding`). */
  readonly voyageCode?: string;

  constructor(
    message: string,
    status: number,
    detail?: string,
    voyageCode?: string,
  ) {
    super(message);
    this.name = "VoyageApiError";
    this.status = status;
    this.detail = detail;
    this.voyageCode = voyageCode;
  }
}

function envTrim(name: string): string | undefined {
  const v = Deno.env.get(name);
  const s = v?.trim();
  return s && s.length > 0 ? s : undefined;
}

/**
 * Embed a single user search string for retrieval (Voyage `input_type: query`).
 * @throws {VoyageApiError} on HTTP errors or empty embedding payload
 */
export async function embedSearchQuery(text: string): Promise<number[]> {
  const apiKey = envTrim("VOYAGE_API_KEY");
  if (!apiKey) {
    throw new VoyageApiError("VOYAGE_API_KEY is not set", 503, undefined, "missing_voyage_key");
  }

  const model = envTrim("VOYAGE_MODEL") ?? DEFAULT_VOYAGE_MODEL;
  const inputType = (envTrim("VOYAGE_INPUT_TYPE") ?? "query") as
    | "query"
    | "document";

  const body: Record<string, unknown> = {
    model,
    input: text,
    input_type: inputType,
    truncation: true,
  };

  const dimRaw = envTrim("VOYAGE_OUTPUT_DIMENSION");
  if (dimRaw && /^\d+$/.test(dimRaw)) {
    body.output_dimension = parseInt(dimRaw, 10);
  }

  const res = await fetch(VOYAGE_EMBEDDINGS_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  const rawText = await res.text();
  let parsed: unknown = {};
  if (rawText) {
    try {
      parsed = JSON.parse(rawText);
    } catch {
      parsed = {};
    }
  }

  if (!res.ok) {
    const detail =
      typeof (parsed as { detail?: unknown })?.detail === "string"
        ? (parsed as { detail: string }).detail
        : rawText.slice(0, 500);
    throw new VoyageApiError(
      detail || res.statusText || "Voyage request failed",
      res.status,
      detail,
      undefined,
    );
  }

  const data = (parsed as { data?: unknown }).data;
  const first = Array.isArray(data) ? data[0] : undefined;
  const emb = (first as { embedding?: unknown })?.embedding;
  if (!Array.isArray(emb) || emb.length === 0) {
    console.error("voyage embeddings: empty data", parsed);
    throw new VoyageApiError(
      "Empty embedding from Voyage",
      502,
      undefined,
      "empty_embedding",
    );
  }

  return emb.map((x: unknown) => Number(x));
}
