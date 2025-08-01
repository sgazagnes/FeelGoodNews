from good_news_pipeline import generate_daily_good_news
import shutil
import os
from datetime import datetime, timezone
import random 
import json 

with open("public/data/personalities.json", "r", encoding="utf-8") as f:
    personality_data = json.load(f)
PERSONALITIES = list(personality_data.keys())

def random_personality_for_today():
    return random.choice(PERSONALITIES)

openai_api_key = os.getenv("OPENAI_API_KEY")
cf_api_token = os.getenv("CLOUDFLARE_API_TOKEN")
cf_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
deepl_api_key = os.getenv("DEEPL_API_KEY")
print("OPENAI_API_KEY present?", bool(os.getenv("OPENAI_API_KEY")), len(openai_api_key))
print("CLOUDFLARE_API_TOKEN present?", bool(os.getenv("CLOUDFLARE_API_TOKEN")), len(cf_api_token))
print("CLOUDFLARE_ACCOUNT_ID present?", bool(os.getenv("CLOUDFLARE_ACCOUNT_ID")), len(cf_account_id))
# print("DEEPL_API_KEY present?", bool(os.getenv("DEEPL_API_KEY")), len(deepl_api_key))
# personality = "gollum"#random_personality_for_today()
# print(f"✨ Today's personality: {personality}")

results = generate_daily_good_news(
    openai_api_key=openai_api_key,
    use_dall_e=False,
    cf_api_token=cf_api_token,
    cf_account_id=cf_account_id,
    deepl_api_key=deepl_api_key,
    personality=None,
    max_articles=4,
    generate_images=True
)

# Save JSON with date and category
# today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
# json_filename = f"public/data/{today_str}_{results['country']}.json"
# os.makedirs("public", exist_ok=True)
# with open(json_filename, "w", encoding="utf-8") as f:
#     json.dump(results, f, ensure_ascii=False, indent=2)

# # Copy images
# os.makedirs("public/images", exist_ok=True)
# for article in results["articles"]:
#     if article["image_url"]:
#         src = article["image_url"]
#         filename = os.path.basename(src)
#         dst = os.path.join("public/images", filename)
#         shutil.copyfile(src, dst)

# print("✅ Daily news generated and ready for Netlify.")
