import csv

input_file = "track_playlist_mapping.csv"
output_file = "track_playlist_mapping_pg.csv"

with open(input_file, newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    fieldnames = ["track_id", "playlist_ids"]

    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:

        playlists = row["playlist_ids"]

        if playlists:
            playlist_array = "{" + ",".join(
                f'"{p.strip()}"' for p in playlists.split("|")
            ) + "}"
        else:
            playlist_array = "{}"

        writer.writerow({
            "track_id": row["track_id"],
            "playlist_ids": playlist_array
        })

print("Converted CSV ready: track_playlist_mapping_pg.csv")
