from dotenv import load_dotenv
load_dotenv()

import json
import os
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

file_path = "tracks_ai_import.jsonl"

batch = []
batch_size = 100

with open(file_path) as f:
    for line in f:
        row = json.loads(line)

        record = {
            "track_id": row["track_id"],
            "primary_genre": row["primary_genre"],
            "secondary_genres": row["secondary_genres"],
            "style_tags": row["style_tags"],
            "mood_keywords": row["mood_keywords"],
            "search_keywords": row["search_keywords"],
            "themes": row["themes"],
            "energy_level": row["energy_level"],
            "summary": row["summary"],
            "combined_tags": row["combined_tags"]
        }

        batch.append(record)

        if len(batch) == batch_size:
            supabase.table("tracks_ai").insert(batch).execute()
            print(f"Inserted {len(batch)} tracks")
            batch = []

if batch:
    supabase.table("tracks_ai").insert(batch).execute()
    print(f"Inserted {len(batch)} tracks")

print("Import finished")
