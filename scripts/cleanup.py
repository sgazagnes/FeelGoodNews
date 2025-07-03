#!/usr/bin/env python3
import os
import re
from datetime import datetime, timedelta, timezone

FOLDER = "public"

now = datetime.now(timezone.utc)
cutoff = now - timedelta(days=7)
# Match files like "2025-07-01_science.json"
pattern = re.compile(r"(\d{4}-\d{2}-\d{2})_.*\.json$")

deleted = []

for filename in os.listdir(FOLDER):
    match = pattern.match(filename)
    if match:
        date_str = match.group(1)
        try:
            # Parse as naive, set tzinfo=UTC for correct comparison
            file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if file_date < cutoff:
                filepath = os.path.join(FOLDER, filename)
                os.remove(filepath)
                deleted.append(filename)
        except ValueError:
            print(f"Skipping file with invalid date: {filename}")

if deleted:
    print("Deleted old files:")
    for f in deleted:
        print(f" - {f}")
else:
    print("No old files to delete.")
