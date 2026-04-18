import os
import csv
from contentful import Client
from dotenv import load_dotenv

load_dotenv()

SPACE_ID = os.getenv("CONTENTFUL_SPACE_ID")
DELIVERY_TOKEN = os.getenv("CONTENTFUL_DELIVERY_TOKEN")

client = Client(
    SPACE_ID,
    DELIVERY_TOKEN,
    timeout_s=10
)

playlists = []
skip = 0
limit = 100

while True:

    entries = client.entries({
        "content_type": "playlist",
        "limit": limit,
        "skip": skip
    })

    if len(entries) == 0:
        break

    for p in entries:

        playlist_id = p.sys["id"]

        title = getattr(p, "title", "")
        description = getattr(p, "description", "")

        tags = getattr(p, "tags", [])

        if tags is None:
            tags = []

        playlists.append({
            "playlist_id": playlist_id,
            "title": title,
            "description": description,
            "tags": "|".join(tags)
        })

    skip += limit


with open("playlists_export.csv", "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=["playlist_id", "title", "description", "tags"]
    )

    writer.writeheader()

    for row in playlists:
        writer.writerow(row)

print(f"Exported {len(playlists)} playlists")
