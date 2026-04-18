import json
import glob

output_files = glob.glob("batch_*_output.jsonl")

merged = []

for file in output_files:
    with open(file, "r") as f:
        for line in f:
            record = json.loads(line)

            custom_id = record["custom_id"]
            entry_id = custom_id.split("-")[1]

            response = record["response"]["body"]["choices"][0]["message"]["content"]
            data = json.loads(response)

            data["entry_id"] = entry_id

            merged.append(data)

print("Total merged records:", len(merged))

with open("merged_ai_tags.json", "w") as f:
    json.dump(merged, f, indent=2)
