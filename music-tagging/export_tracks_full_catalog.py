import os
import csv
from dotenv import load_dotenv
from contentful import Client

load_dotenv()

SPACE_ID = "2fnxgjdy94fy"
ACCESS_TOKEN = os.getenv("CONTENTFUL_DELIVERY_TOKEN")

client = Client(
    SPACE_ID,
    ACCESS_TOKEN,
    timeout_s=10
)

tracks_data = {}

# ------------------------------
# 1. Traverse Artists → Albums → Tracks
# ------------------------------

skip = 0
limit = 100

while True:

    artists = client.entries({
        "content_type": "artist",
        "limit": limit,
        "skip": skip,
        "include": 3
    })

    if not artists:
        break

    for artist in artists:

        artist_name = getattr(artist, "name", "")

        if hasattr(artist, "albums") and artist.albums:

            for album in artist.albums:

                album_title = getattr(album, "title", "")

                if hasattr(album, "tracks") and album.tracks:

                    for track in album.tracks:

                        entry_id = track.sys["id"]
                        title = getattr(track, "title", "")
                        audio_key = getattr(track, "audio_key", "")
                        duration = getattr(track, "duration", "")

                        tracks_data[entry_id] = {
                            "title": title,
                            "artist": artist_name,
                            "album": album_title,
                            "playlist_name": "",
                            "playlist_description": "",
                            "audio_key": audio_key,
                            "duration": duration
                        }

    skip += limit


# ------------------------------
# 2. Traverse Playlists → Tracks
# ------------------------------

skip = 0

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

        playlist_title = getattr(playlist, "title", "")
        playlist_description = getattr(playlist, "description", "")

        if hasattr(playlist, "tracks") and playlist.tracks:

            for track in playlist.tracks:

                entry_id = track.sys["id"]

                if entry_id in tracks_data:

                    tracks_data[entry_id]["playlist_name"] = playlist_title
                    tracks_data[entry_id]["playlist_description"] = playlist_description

    skip += limit


# ------------------------------
# 3. Fetch all tracks to ensure audio_key and duration
# ------------------------------

skip = 0
limit = 1000

while True:

    tracks = client.entries({
        "content_type": "track",
        "limit": limit,
        "skip": skip
    })

    if not tracks:
        break

    for track in tracks:

        entry_id = track.sys["id"]
        title = getattr(track, "title", "")
        duration = getattr(track, "duration", "")
        audio_key = ""
        try:
            audio_key = track.audio_key
        except:
            audio_key = ""

        if entry_id in tracks_data:

            tracks_data[entry_id]["audio_key"] = audio_key
            tracks_data[entry_id]["duration"] = duration

        else:

            tracks_data[entry_id] = {
                "title": title,
                "artist": "",
                "album": "",
                "playlist_name": "",
                "playlist_description": "",
                "audio_key": audio_key,
                "duration": duration
            }

    skip += limit

# ------------------------------
# 4. Write CSV
# ------------------------------

rows = []

for entry_id, data in tracks_data.items():

    rows.append([
        entry_id,
        data["title"],
        data["artist"],
        data["album"],
        data["playlist_name"],
        data["playlist_description"],
        data["audio_key"],
        data["duration"]
    ])

with open("tracks_to_tag_enriched.csv", "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    writer.writerow([
        "entry_id",
        "title",
        "artist",
        "album",
        "playlist_name",
        "playlist_description",
        "audio_key",
        "duration"
    ])

    writer.writerows(rows)

print("Export complete:", len(rows), "tracks")
