import os
import csv
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

rows_updated = 0

with open("tracks_to_tag_enriched.csv", newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        track_id = row["entry_id"]

        data = {
            "track_title": row["title"],
            "artist_name": row["artist"],
            "album_name": row["album"],
            "audio_key": row["audio_key"],
            "duration": row["duration"]
        }

        supabase.table("tracks_ai") \
            .update(data) \
            .eq("track_id", track_id) \
            .execute()

        rows_updated += 1

print("Tracks updated:", rows_updated)

