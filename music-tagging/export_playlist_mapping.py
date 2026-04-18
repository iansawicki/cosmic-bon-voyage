from dotenv import load_dotenv
import os
import csv
from contentful import Client

load_dotenv()

SPACE_ID = os.getenv("CONTENTFUL_SPACE_ID")
DELIVERY_TOKEN = os.getenv("CONTENTFUL_DELIVERY_TOKEN")

client = Client(
    SPACE_ID,
    DELIVERY_TOKEN
)

playlist_map = {}

skip = 0
limit = 100

while True:

    playlists = client.entries({
        "content_type": "playlist",
        "limit": limit,
        "skip": skip,
        "include": 2
    })

    if not playlists:
        break

    for playlist in playlists:

        playlist_title = playlist.title

        if hasattr(playlist, "tracks"):

            for track in playlist.tracks:

                track_id = track.sys["id"]

                if track_id not in playlist_map:
                    playlist_map[track_id] = []

                playlist_map[track_id].append(playlist_title)

    skip += limit


with open("track_playlist_mapping.csv", "w", newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["track_id", "playlist_ids"])

    for track_id, playlists in playlist_map.items():
        writer.writerow([track_id, "|".join(playlists)])


print(f"Exported mapping for {len(playlist_map)} tracks")
