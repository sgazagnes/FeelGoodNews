#!/usr/bin/env python3
import os
import re
import json
from datetime import datetime, timedelta, timezone

FOLDER = "public/data/"
IMAGE_PREFIX = "public/images/"

now = datetime.now(timezone.utc)
cutoff = now - timedelta(days=7)

# Match files like "2025-07-01_science.json"
pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_.*\.json$")

deleted_jsons = []
deleted_images = []

for filename in os.listdir(FOLDER):
    match = pattern.match(filename)
    if match:
        date_str = match.group(1)
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if file_date < cutoff:
                filepath = os.path.join(FOLDER, filename)

                # Load JSON and find image paths
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for article in data.get("articles", []):
                            image_path = article.get("image_url", "")
                            if image_path.startswith(IMAGE_PREFIX) and os.path.exists(image_path) and 'fallback' not in image_path:
                                os.remove(image_path)
                                deleted_images.append(image_path)
                except Exception as e:
                    print(f"Error reading or parsing {filename}: {e}")

                # Delete the JSON file
                os.remove(filepath)
                deleted_jsons.append(filename)

        except ValueError:
            print(f"Skipping file with invalid date: {filename}")

if deleted_jsons:
    print("Deleted old JSON files:")
    for f in deleted_jsons:
        print(f" - {f}")
else:
    print("No old JSON files to delete.")

if deleted_images:
    print("Deleted associated images:")
    for img in deleted_images:
        print(f" - {img}")
else:
    print("No associated images to delete.")
