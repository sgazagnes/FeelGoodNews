from good_news_pipeline import generate_daily_good_news
import shutil
import os
from datetime import datetime
import random 

PERSONALITIES = [
    "darth_vader",
    "gary_lineker",
    "drunk_philosopher",
    "shakespeare",
    "gordon_ramsay"
]

def random_personality_for_today():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    random.seed(today)
    return random.choice(PERSONALITIES)

api_key = os.getenv("OPENAI_API_KEY")

personality = random_personality_for_today()
print(f"✨ Today's personality: {personality}")

results = generate_daily_good_news(
    api_key=api_key,
    country="global",
    personality=personality,
    max_articles=3,
    generate_images=False
)

# Save JSON
os.makedirs("public", exist_ok=True)
with open("public/latest_news.json", "w", encoding="utf-8") as f:
    import json
    json.dump(results, f, ensure_ascii=False, indent=2)

# Copy images
os.makedirs("public/images", exist_ok=True)
for article in results["articles"]:
    if article["image_url"]:
        src = article["image_url"]
        filename = os.path.basename(src)
        dst = os.path.join("public/images", filename)
        shutil.copyfile(src, dst)

print("✅ Daily news generated and ready for Netlify.")
