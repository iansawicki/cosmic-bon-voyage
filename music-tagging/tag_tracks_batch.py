import os
import json
import math
import pandas as pd
from openai import OpenAI
from contentful_management import Client
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CONTENTFUL_MANAGEMENT_TOKEN = os.getenv("CONTENTFUL_MANAGEMENT_TOKEN")
SPACE_ID = "2fnxgjdy94fy"
ENVIRONMENT_ID = "master"
CONTENT_TYPE_ID = "track"

# Fields to update in Contentful
FIELDS_TO_UPDATE = ["genres", "mood_keywords", "themes", "search_keywords", "summary"]

# ── Clients ────────────────────────────────────────────
client = OpenAI(api_key=OPENAI_API_KEY)
cf_client = Client(CONTENTFUL_MANAGEMENT_TOKEN)
environment = cf_client.environments(SPACE_ID).find(ENVIRONMENT_ID)

# ── JSON Schema ─────────────────────────────────────────
SCHEMA = {
    "type": "object",
    "properties": {

        "themes": {
            "type": "object",
            "properties": {
                "flow": {
                    "type": "object",
                    "properties": {
                        "assigned": {"type": "boolean"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["assigned", "confidence"],
                    "additionalProperties": False
                },
                "ritual": {
                    "type": "object",
                    "properties": {
                        "assigned": {"type": "boolean"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["assigned", "confidence"],
                    "additionalProperties": False
                },
                "expanded_state": {
                    "type": "object",
                    "properties": {
                        "assigned": {"type": "boolean"},
                        "confidence": {"type": "number"}
                    },
                    "required": ["assigned", "confidence"],
                    "additionalProperties": False
                }
            },
            "required": ["flow", "ritual", "expanded_state"],
            "additionalProperties": False
        },

        "primary_genre": {
            "type": "string"
        },

        "secondary_genres": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 2
        },

        "style_tags": {
            "type": "array",
            "items": {"type": "string"}
        },

        "mood_keywords": {
            "type": "array",
            "items": {"type": "string"}
        },

        "search_keywords": {
            "type": "array",
            "items": {"type": "string"}
        },

        "energy_level": {
            "type": "number"
        },

        "summary": {
            "type": "string"
        }

    },

    "required": [
        "themes",
        "primary_genre",
        "secondary_genres",
        "style_tags",
        "mood_keywords",
        "search_keywords",
        "energy_level",
        "summary"
    ],

    "additionalProperties": False
}

# ── Prompt ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert music curator for a curated internet radio app centered on three core states: flow, ritual, expanded states.

These themes represent experiential states that listeners may enter while engaging with the music.

Definitions:

RITUAL:
Slow, intentional, ceremonial or repetitive practices that create a sense of sacred space, grounding, or presence. Often associated with mindful routines or reflective moments.

Examples:
sleep, meditation, tea ceremony, bathing, intimacy, journaling, prayer, walking in nature, contemplative reading, gentle movement, yoga, breath awareness.

FLOW:
A state of deep concentration and effortless engagement where attention is fully absorbed in an activity. Associated with productivity, creativity, learning, and sustained focus.

Examples:
focused work, studying, writing, coding, creative production, problem-solving, designing, reading with concentration, sustained mental effort.

EXPANDED STATES:
Altered or heightened states of consciousness where perception, awareness, or sense of self may shift beyond ordinary waking experience.

Examples:
psychedelic experiences, breathwork journeys, deep meditation, lucid dreaming, trance states, ecstatic dance, sensory immersion, deep sleep or dream states.

Given this track metadata:
- Title: {title}
- Artist: {artist}
- Album: {album}
- Playlist: {playlist_name}
- Mood description: {playlist_description}

Before assigning themes, evaluate how the track might function in a listening context.

Consider:
- listener state
- tempo and intensity
- emotional tone
- environment where the music might be used
- playlist context

Then determine the most appropriate experiential states.

Your tasks:

1. Evaluate the likelihood of each experiential state (flow, ritual, expanded_state).

For each theme determine:
- whether the state is likely to occur
- a confidence score between 0 and 1

All three themes must always be present.

Do not assign all themes equally. Prefer the one or two most strongly associated experiential states.

2. Determine the PRIMARY GENRE of the track.
This should represent the main musical family.

Examples:
Ambient, Electronic, Experimental, Downtempo, Neoclassical, Jazz, Soundtrack.

3. Determine up to TWO SECONDARY GENRES.
These should be more specific stylistic genres related to the primary genre.

Examples:
Drone, Dark Ambient, Minimal, Psychedelic Ambient, Electroacoustic.

4. Generate 5–10 STYLE TAGS describing sonic or production characteristics.

Examples:
Minimal, Hypnotic, Textural, Atmospheric, Repetitive, Layered, Organic, Field Recordings.

5. Generate 8–15 MOOD KEYWORDS describing the emotional or experiential character of the track.

Examples:
Meditative, Serene, Contemplative, Expansive, Dreamlike.

6. Generate 5–10 SEARCH KEYWORDS that help users discover the track.

7. Estimate an ENERGY LEVEL between 0.0 and 1.0 representing musical intensity.

Energy scale:
0.0 → extremely calm / ambient
0.3 → meditative / slow
0.5 → steady groove
0.7 → energetic
1.0 → intense / driving

8. Write a concise summary of the track.

Rules:

- Primary genre must be a real music genre.
- Secondary genres should refine the primary genre.
- Avoid inventing genres.
- Use standard capitalization.
- Style tags should describe sound design or structure.
- Mood keywords should describe emotional tone.

Output ONLY valid JSON matching the schema. Do not include explanations.
"""

# ── Load CSV ─────────────────────────────────────────────
df = pd.read_csv("tracks_to_tag.csv")

# ── Config for batching
BATCH_SIZE = 500
total_tracks = len(df)
num_batches = math.ceil(total_tracks / BATCH_SIZE)

print(f"Total tracks: {total_tracks}, creating {num_batches} batch file(s).")

# ── Process batches ──────────────────────────────────────
for batch_num in range(num_batches):

    start_idx = batch_num * BATCH_SIZE
    end_idx = min((batch_num + 1) * BATCH_SIZE, total_tracks)
    batch_df = df.iloc[start_idx:end_idx]

    requests = []

    for idx, row in batch_df.iterrows():
        entry_id = row["entry_id"]
        custom_id = f"track-{entry_id}-{idx}"

        title = str(row.get("title", "") or "")
        artist = str(row.get("artist", "") or "")
        album = str(row.get("album", "") or "")
        playlist_name = str(row.get("playlist_name", "") or "")
        playlist_description = str(row.get("playlist_description", "") or "")

        user_content = (
            f"Title: {title}\n"
            f"Artist: {artist}\n"
            f"Album: {album}\n"
            f"Playlist: {playlist_name}\n"
            f"Mood description: {playlist_description}"
        )

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 500,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "music_tagging_response",
                        "strict": True,
                        "schema": SCHEMA
                    }
                },
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                            .replace("{title}", title)
                            .replace("{artist}", artist)
                            .replace("{album}", album)
                            .replace("{playlist_name}", playlist_name)
                            .replace("{playlist_description}", playlist_description)
                    },
                    {
                        "role": "user",
                        "content": user_content
                    }
                ]
            }
        }

        requests.append(request)

    # ── Save JSONL
    batch_file_path = f"batch_input_{batch_num+1}.jsonl"
    with open(batch_file_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Batch {batch_num+1}/{num_batches} saved: {len(requests)} tracks → {batch_file_path}")

    # ── Upload batch
    try:
        with open(batch_file_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")

        print(f"Uploaded batch {batch_num+1}: File ID {batch_file.id}")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )

        print(f"Batch job created: ID {batch_job.id}")

        import time
        time.sleep(8)

    except Exception as e:
        print(f"Batch {batch_num+1} failed:", str(e))
