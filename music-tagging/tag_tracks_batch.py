import os
import json
import math
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI
from contentful_management import Client
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tagging.prompt import (  # noqa: E402
    SCHEMA,
    openai_batch_request_body,
)

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

# SCHEMA imported from tagging.prompt

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

        body = openai_batch_request_body(
            title=title,
            artist=artist,
            album=album,
            playlist_name=playlist_name,
            playlist_description=playlist_description,
        )

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
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
