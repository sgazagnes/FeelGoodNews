# FeelGoodNews

FeelGoodNews is a Python application that automatically fetches science and technology news, analyzes each article with an LLM to detect *uplifting, positive stories*, generates AI-based summaries in different personalities, and optionally creates cheerful images to illustrate the news.

**The goal: bring more good news to people's daily feeds.**

This is now available as a mobile app for Android users! https://play.google.com/store/apps/details?id=com.simongazagnes.feelgoodnews&hl=en_US
---

## ✨ Features

- 📰 **RSS Scraper**
  - Fetches articles from selected science and environment feeds (e.g., ScienceDaily)
- 🤖 **LLM Analysis**
  - Detects whether a story is "good news"
  - Assigns a positivity sentiment score (0–1)
  - Categorizes the article
  - Provides reasoning and key positive elements
- 🎨 **Image Generation**
  - Creates custom prompts for AI image generators

---

## 🛠️ Installation for local use

1. Clone this repository:
```bash
git clone https://github.com/sgazagnes/feelgoodnews.git
cd feelgoodnews
```

2. Install the dependencies:
```bash
pip install -r requirements.txt
```

Dependencies include:
- `openai`
- `feedparser`
- `requests`
- `Pillow`
- `cloudflare`
- `json5`
- etc.

---

## ⚙️ Configuration

The app requires API keys for:
- OpenAI API (text analysis and optional DALL·E images)
- Cloudflare Workers AI (optional image generation)

You can set them as environment variables:
```bash
export OPENAI_API_KEY="your_openai_key"
export CF_API_TOKEN="your_cloudflare_token"
export CF_ACCOUNT_ID="your_cloudflare_account_id"
```

Or pass them directly in code:
```python
from feelgoodnews import generate_daily_good_news

generate_daily_good_news(
    openai_api_key="your_openai_key",
    use_dall_e=True,
    cf_api_token="your_cloudflare_token",
    cf_account_id="your_cloudflare_account_id",
    personality="shakespeare",
    max_articles=10,
    generate_images=True
)
```

---

## 🚀 Usage

The main entry point is the `generate_daily_good_news()` function.

**Example script:**
```python
from feelgoodnews import generate_daily_good_news

generate_daily_good_news(
    openai_api_key="YOUR_API_KEY",
    use_dall_e=True,
    cf_api_token="YOUR_CF_TOKEN",
    cf_account_id="YOUR_CF_ACCOUNT_ID",
    personality="gordon_ramsay",
    max_articles=5,
    generate_images=True
)
```

**Outputs:**
- JSON files in `public/data/`
- Images in `public/images/`

Each JSON file includes:
- Article metadata
- Sentiment analysis
- Generated summaries
- Image prompt & saved image path


## 📄 License

MIT License. See LICENSE.
