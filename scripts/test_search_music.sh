#!/usr/bin/env bash
# Test the search-music Edge Function (local `supabase functions serve` or deployed project).
#
# Prerequisites:
#   - jq (brew install jq)
#   - Config: copy configs/*_supa_db.json.example → configs/dev_supa_db.json (or prod) and fill values.
#
# By default reads configs/dev_supa_db.json for supabase_url + supabase_key (service role or anon JWT).
# Override with env vars if needed (they win over the JSON file).
#
#   CONFIG_PROFILE=prod   — use configs/prod_supa_db.json instead of dev
#   CONFIG_FILE=/path.json — explicit JSON path (wins over CONFIG_PROFILE)
#
# For LOCAL serve: in another terminal:
#     supabase functions serve search-music --env-file .env
#   Then set SUPABASE_URL=http://127.0.0.1:54321 and SUPABASE_ANON_KEY from `supabase status`
#   (those env vars override the config file).
#
# Usage:
#   ./scripts/test_search_music.sh psychedelic ambient
#   CONFIG_PROFILE=prod ./scripts/test_search_music.sh chill vibes

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  head -n 28 "$0" | tail -n +2 | sed 's/^# \{0,1\}//'
  exit 0
fi

if ! command -v jq &>/dev/null; then
  echo "jq is required (install: brew install jq)" >&2
  exit 1
fi

PROFILE="${CONFIG_PROFILE:-dev}"
CONFIG_JSON="${CONFIG_FILE:-$ROOT/configs/${PROFILE}_supa_db.json}"

if [[ ! -f "$CONFIG_JSON" ]]; then
  echo "Missing config file: $CONFIG_JSON" >&2
  echo "Create it from configs/${PROFILE}_supa_db.json.example or set CONFIG_FILE." >&2
  exit 1
fi

# Env wins over JSON (so you can point at local serve without editing JSON).
if [[ -z "${SUPABASE_URL:-}" ]]; then
  SUPABASE_URL="$(jq -r '.supabase_url // empty' "$CONFIG_JSON")"
fi
if [[ -z "${SUPABASE_URL:-}" || "$SUPABASE_URL" == "null" ]]; then
  echo "supabase_url is empty in $CONFIG_JSON (or set SUPABASE_URL)." >&2
  exit 1
fi

KEY="${SUPABASE_ANON_KEY:-${ANON_KEY:-${SUPABASE_KEY:-}}}"
if [[ -z "$KEY" ]]; then
  KEY="$(jq -r '.supabase_key // empty' "$CONFIG_JSON")"
fi
if [[ -z "$KEY" || "$KEY" == "null" ]]; then
  echo "No API key: set supabase_key in $CONFIG_JSON or export SUPABASE_ANON_KEY / ANON_KEY / SUPABASE_KEY." >&2
  exit 1
fi

if [[ -n "${SEARCH_MUSIC_URL:-}" ]]; then
  FUNC_URL="$SEARCH_MUSIC_URL"
else
  FUNC_URL="${SUPABASE_URL%/}/functions/v1/search-music"
fi

QUERY="${*:-chill ambient for focus}"
BODY=$(jq -nc --arg q "$QUERY" '{query: $q}')

RESP=$(mktemp)
trap 'rm -f "$RESP"' EXIT

echo "config: $CONFIG_JSON" >&2
echo "POST $FUNC_URL" >&2
echo "query: $QUERY" >&2
echo "" >&2

code="$(
  curl -sS -o "$RESP" -w "%{http_code}" \
    -X POST "$FUNC_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $KEY" \
    -H "apikey: $KEY" \
    -d "$BODY"
)"

jq . <"$RESP" 2>/dev/null || cat "$RESP"
echo "" >&2
echo "HTTP $code" >&2

if [[ "$code" != "200" ]]; then
  exit 1
fi
