import requests
import feedparser
import json
import json5
import re
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional
import time
from urllib.parse import urljoin
import random
import openai
from openai import OpenAI
import os
import base64
from io import BytesIO
from PIL import Image
import hashlib
from cloudflare import Cloudflare
from collections import defaultdict
import deepl
from bs4 import BeautifulSoup
from newspaper import Article
import numpy as np
from deep_translator import GoogleTranslator
from newspaper import network

# network.USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
from newspaper import Config

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'

# add your proxy information

config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10

SIMILARITY_THRESHOLD = 0.8

def extract_full_article(url):
    try:
        article = Article(url.strip())
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Failed to extract article with newspappers from {url}: {e}")
        return extract_full_article_fallback(url)
    #     print(f"Failed to extract article from {url}")
    #     return ""

def extract_full_article_fallback(url):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
        }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        content_section = soup.find('section', class_='article__content')
        if content_section:
            paragraphs = content_section.find_all('p')
            return '\n'.join(p.get_text() for p in paragraphs)
        else:
            return ""
    except Exception as e:
        print(f"Twice Failed to extract article from {url}: {e}")
        print("")
        return ""
    

def strip_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text()

def extract_article_text(entry):
    """Return clean full content if available, fallback to summary."""
    print(entry)
    if 'content' in entry and entry.content:
        raw = entry.content[0].get('value', '') or ''

    if 'summary' in entry and entry.summary:
        raw2 = entry.get('summary', '') or ''
    return strip_html(raw), strip_html(raw2)

def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()



def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_recent_dates(days_back):
    today = datetime.now().date()
    return [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_back)]


@dataclass
class NewsArticle:
    title: str
    summary: str
    content: str
    url: str
    source: str
    published: datetime
    category: str
    sentiment_score: float = 0.0
    is_good_news: bool = False
    reasoning: str = ""
    image_url: str = ""
    image_prompt: str = ""
    embedding: list = None

class LLMAnalyzer:
    """Handle all LLM interactions for sentiment analysis, personality generation, and image prompts"""
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4.1-mini", use_dall_e: bool = False, cf_api_token: str = None, cf_account_id: str = None):

        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        # response = self.client.models.list()
        # print("Models available:", [m.id for m in response.data])
        self.model = model
        self.use_dall_e = use_dall_e

        self.cf_client = Cloudflare(api_token=cf_api_token) if cf_api_token else None
        self.cf_account_id = cf_account_id if cf_account_id else None
        
    def get_embedding(self, text: str):
        # print("Generating embedding for text:", text[:50], "...")
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            # print("Generated embedding for text")

            return response.data[0].embedding
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
        
    def analyze_news_sentiment(self, title: str, summary:str, content: str) -> Dict:
        """Analyze if news is positive and get sentiment score using LLM"""
        
        prompt = f"""
        Analyze this news article and determine if it's "good news" that would make most people feel positive and hopeful. Give it a sentiment score between 0 and 1, where 1 means it is really uplifting and positive, and is likely to interest a wide audience.

        TITLE: {title}
        SUMMARY: {summary}
        CONTENT: {content}

        Good news criteria:
        - Stories about people helping others, acts of kindness, charity
        - Scientific breakthroughs, medical advances, cures
        - Environmental progress, conservation successes
        - Educational achievements, scholarships, literacy programs
        - Community coming together, cooperation, unity
        - Individual achievements that inspire others
        - Government policies that benefit citizens
        - Economic improvements for ordinary people
        - Cultural celebrations, positive milestones
        - Animals being rescued or protected
        - Technology solving real problems
        - Recovery stories, overcoming challenges

        NOT good news:
        - Crime, violence, accidents, disasters
        - Political conflicts, wars, protests
        - Economic downturns, layoffs
        - Health crises, disease outbreaks
        - Environmental destruction
        - Celebrity scandals or gossip
        - Negative political news
        - Corporate controversies

        Then, select the **single most appropriate category** for this news story. You must choose **exactly one** of the following categories:

        - Health
        - Environment
        - Technology
        - Human Rights
        - Space
        - Other

        Do not invent any other categories. Only respond with one of these exact names.


        Respond in JSON format:
        {{
            "is_good_news": true/false,
            "sentiment_score": 0.00-1.00,
            "category": "Health, Environment, Technology, Human Rights, Space or Other",
            "reasoning": "Brief explanation of why this is or isn't good news",
            "key_positive_elements": ["list", "of", "positive", "aspects"],
            "emotional_impact": "uplifting/inspiring/heartwarming/hopeful/etc"
        }}
        """
        
        if self.client:
            # try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing news sentiment and identifying positive, uplifting stories."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            result = json5.loads(response.choices[0].message.content)
            return result
            # except Exception as e:
            #     print(f"LLM Analysis Error: {e}")
            #     return self._fallback_analysis(title, content)
        else:
            return prompt
    
    def _fallback_analysis(self, title: str, content: str) -> Dict:
        text = f"{title} {content}".lower()

        positive_indicators = [...]
        negative_indicators = [...]

        positive_count = ...
        negative_count = ...

        is_good = ...
        score = ...
        
        # Naive category assignment
        if any(word in text for word in ['health', 'medical', 'hospital']):
            category = "Health"
        elif any(word in text for word in ['environment', 'climate', 'nature']):
            category = "Environment"
        elif any(word in text for word in ['technology', 'tech', 'innovation']):
            category = "Technology"
        elif any(word in text for word in ['human rights', 'equality', 'justice']):
            category = "Human Rights"
        elif any(word in text for word in ['space', 'astronomy', 'stars']):
            category = "Space"
        else:
            category = "Other"

        return {
            "is_good_news": is_good,
            "sentiment_score": score,
            "category": category,
            "reasoning": f"Found {positive_count} positive and {negative_count} negative indicators",
            "key_positive_elements": ["fallback analysis"],
            "emotional_impact": "positive" if is_good else "neutral"
        }

    
    def generate_image_prompt(self, article: NewsArticle) -> str:
        """Generate DALL-E image prompt based on article content"""
        
        if(self.use_dall_e):
            prompt = f"""
            Based on this good news article, create a detailed image prompt for DALL-E that would generate an uplifting, positive image that represents the story.

            TITLE: {article.title}
            CONTENT: {article.content}
            POSITIVE ELEMENTS: {article.reasoning}

            Guidelines for the image prompt:
            - Make it warm, uplifting, and positive
            - Make it as realistic as possible
            - Focus on the emotion and feeling of the story
            - Use cheerful but natural colors
            - Keep it family-friendly and universally positive
            - Make it photorealistic
            - Avoid text or words in the image

            Create a detailed image prompt (1-2 sentences max) that captures the essence and positive emotion of this news story:
            """
        else:
            prompt = f"""
            Based on the following good news article, create a detailed image prompt for a text-to-image LLM that will generate an image that clearly represents this story.

            TITLE: {article.title}
            CONTENT: {article.content}

            **First**, extract 1–3 key subjects, objects, or themes from the article that are central to what happened (e.g., the people, animals, locations, or objects involved).

            **Second**, describe how these elements could be shown symbolically or through metaphors that are directly relevant to the article.

            **Third**, write a **single image prompt (1–2 sentences max)** that:
            - Is warm and positive
            - Makes sure the final image is a detailed drawing or painting (not photorealistic)
            - Includes only natural and simple colors
            - Is family-friendly
            - Does NOT depict real human figures (use symbolic elements, silhouettes, objects, animals, landscapes, or abstract scenes)
            - Avoids any text or words
            - **Clearly depicts the extracted subjects or themes without generic symbols (like trees, butterflies, or sunshine) unless they are explicitly part of the story**


            **Provide only the final Image Prompt after doing this reasoning internally.**
            """
        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at creating detailed, positive image prompts for LLMs."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=250
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Image Prompt Generation Error: {e}")
                return self._fallback_image_prompt(article)
            
        else:
            return self._fallback_image_prompt(article)
    
    def generate_image_cf(self, prompt: str, article_title: str) -> Optional[str]:
        """
        Generate an image from prompt using Cloudflare Workers AI, or skip if file exists.
        """
        # Generate safe filename
        safe_title = re.sub(r'[^\w\s-]', '', article_title).strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)[:50]
        os.makedirs('public/images', exist_ok=True)
        filename = f"public/images/news_image_{safe_title}.png"

        # Check if the file already exists
        if os.path.exists(filename):
            print(f"✅ Image already exists: {filename}")
            return filename

        # If not, generate the image
        try:
            data = self.cf_client.ai.with_raw_response.run(
                "@cf/black-forest-labs/flux-1-schnell",
                account_id=self.cf_account_id,
                prompt=prompt,
            ).json()

            image_base64 = data["result"]["image"]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)

            # Load image into PIL
            image = Image.open(BytesIO(image_bytes))

            # Resize to 512x512
            image_resized = image.resize((512, 512), Image.LANCZOS)

            # Ensure directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # Save resized image
            image_resized.save(filename, format="PNG")

            print(f"✅ Image saved to: {filename}")
            return filename

        except Exception as e:
            print(f"Error generating image: {e}")
            return None



    def _fallback_image_prompt(self, article: NewsArticle) -> str:
        """Fallback image prompts when LLM is not available"""
        text = f"{article.title} {article.content}".lower()
        
        if any(word in text for word in ['science', 'research', 'breakthrough', 'discovery']):
            return "A bright, modern laboratory with scientists working together, warm lighting, hopeful atmosphere, photorealistic"
        elif any(word in text for word in ['community', 'help', 'volunteer', 'charity']):
            return "People helping each other in a warm, sunny community setting, diverse group, smiling silhouettes, golden hour lighting"
        elif any(word in text for word in ['environment', 'nature', 'conservation']):
            return "Beautiful, thriving natural landscape with clear blue sky, green trees, and clean water, photorealistic, inspiring"
        elif any(word in text for word in ['education', 'school', 'learning']):
            return "Bright, welcoming classroom or library with books and learning materials, warm lighting, inspiring atmosphere"
        else:
            return "Uplifting scene with warm golden light, people celebrating or achieving something positive, photorealistic, inspiring"
    
    def generate_image_de(self, prompt: str, article_title: str) -> Optional[str]:
        """Generate image using DALL-E and save locally"""
        if not self.use_openai or not self.client:
            print("OpenAI API not available for image generation")
            return None
        
        try:
            print(f"🎨 Generating image with prompt: {prompt[:200]}...")
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="512x512",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            
            # Download and save the image locally
            img_response = requests.get(image_url)
            if img_response.status_code == 200:
                # Create filename based on article title
                safe_title = re.sub(r'[^\w\s-]', '', article_title).strip()
                safe_title = re.sub(r'[-\s]+', '-', safe_title)[:50]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create images directory if it doesn't exist
                os.makedirs('public/images', exist_ok=True)
                
                filename = f"public/images/news_image_{safe_title}.png"
                
                with open(filename, 'wb') as f:
                    f.write(img_response.content)
                
                print(f"✅ Image saved: {filename}")
                return filename
            else:
                print(f"Failed to download image: {img_response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def generate_summary_response(self, article: NewsArticle, personality: str) -> str:
        """Generate news presentation using LLM"""
        
        
        prompt = f"""
        Present this good news story in your style, using clear, simple, and friendly language. Avoid jargon or complicated words. Write in a warm, storytelling tone, but focusing on the news topic and details, without unnecessary fillers. When summarizing actions, policies, or tips that led to the positive outcome, **briefly explain what they are and how they helped**, so the reader can understand. Keep each section concise and informative.

        Write the news as a structured text using the following sections. **Each section must start with the section title in bold (using double asterisks) and normal capitalization.** Then write a short paragraph. Leave an empty line between sections.

        Here are the sections:

        **Context**
        **What happened**
        **Impact**
        **What's next step**
        **One-sentence takeaway**

        Once you finished, translate the text into french first, and then spaning. At the end, do not add any other commentary or formatting.

        Respond ONLY in JSON format like this:
        {{
        "title": "A short, catchy headline in your style (capitalize only the first letter)",
        "text": "All the sections above, separated by line breaks."
        "title_fr": "The title translated into French",
        "text_fr": "French translation of the sections above",
        "title_es": "The title translated into Spanish",
        "text_es": "Spanish translation of the sections above"
        }}

        Do not include any markdown or code fences.

        NEWS TITLE: {article.title}
        NEWS SUMMARY: {article.summary}
        NEWS CONTENT: {article.content[:500]}
        POSITIVE ELEMENTS: {article.reasoning}
        """

        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                 model=self.model,
                    messages=[
                        {"role": "system", "content": f" You are a journalist specializing in positive news stories. You excel at simplifying complex topics and making them engaging and easy to understand. Your audience consists of busy people who only have a few minutes each day to stay informed. Your goal is to convey the most important details clearly, in a warm, and easy-to-read style."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                if content.startswith('"""'):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit('"""', 1)[0]

                result = json5.loads(content)
                # print(result)
                return result
                
            except Exception as e:
                print(f"Generation Error: {e}")
                return None
        else:
            return None
    def generate_personality_response(self, article: NewsArticle, personality: str) -> str:
        """Generate personality-based news presentation using LLM"""
        
        with open("public/data/personalities.json", "r", encoding="utf-8") as f:
            personality_prompts = json5.load(f)
        
        if personality not in personality_prompts:
            personality = 'darth_vader'
        
        char_info = personality_prompts[personality]
        
        prompt = f"""
        You are {personality.replace('_', ' ').title()}, {char_info['description']}.

        CHARACTER TRAITS: {char_info['traits']}

        Present this good news story in your style, using clear, simple, and friendly language. Avoid jargon or complicated words. Write in a warm, storytelling tone.

        Write the news as a structured text using the following sections. **Each section must start with the section title in bold (using double asterisks) and normal capitalization.** Then write one or two short sentences. Leave an empty line between sections.

        Here are the sections:

        **Context**
        **What happened**
        **Impact**
        **What's next**
        **One-sentence takeaway**

        At the end, do not add any other commentary or formatting.

        Respond ONLY in JSON format like this:
        {{
        "title": "A short, catchy headline in your style (capitalize only the first letter)",
        "text": "All the sections above, separated by line breaks."
        }}

        Do not include any markdown or code fences.

        NEWS TITLE: {article.title}
        NEWS CONTENT: {article.content}
        POSITIVE ELEMENTS: {article.reasoning}
        """

        
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are {personality.replace('_', ' ').title()} presenting good news. Stay in character completely."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                if content.startswith('"""'):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit('"""', 1)[0]

                result = json5.loads(content)
                # print(result)
                return result
                
            except Exception as e:
                print(f"Personality Generation Error: {e}")
                return self._fallback_personality(article, personality)
        else:
            return self._fallback_personality(article, personality)
    
    def _fallback_personality(self, article: NewsArticle, personality: str) -> str:
        """Fallback personality responses when LLM is not available"""
        templates = {
            'darth_vader': f"*Heavy breathing* Young padawan, this news brings balance to the Force: {article.title}\n\nThe Empire of good deeds reports: {article.content}\n\nMost impressive... most impressive indeed.",
            'gary_lineker': f"What an absolute cracker of a story! {article.title}\n\nHere's the beautiful play-by-play: {article.content}\n\nSensational stuff, truly sensational!",
            'drunk_philosopher': f"*hiccup* Listen mate, this is actually beautiful... {article.title}\n\n*sway* And here's the profound bit: {article.content}\n\n*raises glass* To humanity! *hiccup*",
            'shakespeare': f"Hark! What noble tale doth grace our ears: {article.title}\n\nWherein we learn of human kindness: {article.content}\n\nAll's well that ends well, and this tale ends most fair.",
            'gordon_ramsay': f"Bloody hell, this is beautiful! {article.title}\n\nLook at this gorgeous story: {article.content}\n\nAbsolutely stunning! That's what I call a perfect dish of humanity!"
        }
        return templates.get(personality, templates['darth_vader'])
    
    def generate_category_summary(self, articles, category: str) -> str:
        """Generate a short summary of all good news in a category"""
        # titles = "\n".join(f"- {a.title}" for a in articles)
        titles = "\n".join(f"- {a['title']}" for a in articles)
        prompt = f"""
        Summarize the following {category} good news headlines into one sentence highlighting the main themes of today's news.
        Headlines:
        {titles}

        Do not list the titles again. Do not include information that is not necessary or generic fillers.
        """
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You summarize groups of positive news stories in a positive and concise way."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Category summary error: {e}")
                return ""
        return ""


class GoodNewsScraper:
    def __init__(self, llm_api_key: str = None, use_dall_e: bool = False, cf_api_token: str = None, cf_account_id: str = None):
        self.llm_analyzer = LLMAnalyzer(openai_api_key=llm_api_key, use_dall_e=use_dall_e, cf_api_token=cf_api_token, cf_account_id=cf_account_id)
        
        self.news_sources = [
            # Global general news
            # "https://feeds.bbci.co.uk/news/rss.xml",
            "https://feeds.bbci.co.uk/news/technology/rss.xml?edition=uk",
            "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml?edition=uk",
            "https://www.sciencedaily.com/rss/top.xml",
            "https://www.nature.com/nature.rss",
            "https://feeds.feedburner.com/ConservationInternationalBlog",
            "https://www.nasa.gov/rss/dyn/breaking_news.rss",
            "http://earth911.com/feed/",
            "https://grist.org/feed/",
            "https://www.hrw.org/rss/news",
            "https://hrp.law.harvard.edu/feed",
            "https://www.hhrjournal.org/category/blog/feed/",
            "https://www.newscientist.com/feed/home/?cmpid=RSS%7CNSNS-Home",
            "https://www.france24.com/en/earth/rss",
            "https://www.france24.com/en/business/rss",
            "https://www.france24.com/en/culture/rss",
            "https://www.france24.com/en/earth/rss",
            "https://www.france24.com/en/health/rss"
            # "https://www.lemonde.fr/en/culture/rss_full.xml",
            # "https://www.lemonde.fr/en/environment/rss_full.xml",
            # "https://www.lemonde.fr/en/climate-change/rss_full.xml",


        ]

        self.previous_articles = self.load_previous_articles()

    def load_previous_articles(self) -> List[Dict]:
                # --- Load Articles --[-
        articles= []
        print("📦 Loading articles...")
        for fname in os.listdir("public/data"):
            if not fname.endswith(".json"):
                continue
            try:
                date_part = fname.split("_")[0]
                if date_part in get_recent_dates(2):
                    with open(os.path.join("public/data", fname), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        for article in data.get("articles", []):
                            articles.append({
                                "summary": normalize_text(article["summary"]),
                                "title": article["title"],
                                "category": data.get("category", "Unknown"),
                                "url": article.get("url", ""),
                                "embedding": article.get("embedding", "") 
                            })

            except Exception as e:
                print(f"Error reading {fname}: {e}")     

        return articles

    def fetch_rss_feed(self, url: str) -> List[Dict]:
        """Fetch and parse RSS feed"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; MyRSSReader/1.0)"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # raise exception for HTTP errors

            feed = feedparser.parse(response.content)

            print("Feed parsing status:", feed.bozo)
            if feed.bozo:
                print("Parsing error:", feed.bozo_exception)

            articles = []
            for entry in feed.entries[:20]:  # Limit to recent articles
                # clean_content, clean_summary = extract_article_text(entry)
                full_content = extract_full_article(entry.get('link', ''))
                if(full_content == ""):
                    full_content = strip_html(entry.get('summary', ''))
                articles.append({
                    'title': entry.get('title', ''),
                    'summary': strip_html(entry.get('summary', '')),
                    'content': full_content,
                    'link': entry.get('link', ''),
                    'published': entry.get('published_parsed', None),
                    'source': feed.feed.get('title', url)
                })

            return articles
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []

    def is_recent_article(self, published_date: datetime, days_back: int = 1) -> bool:
        """Check if article was published within the specified number of days"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=days_back)
        return published_date.date() >= yesterday
    
    def is_yesterday_article(self, published_date: datetime) -> bool:
        """Check if article was published yesterday (00:00 to 23:59)"""
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        return published_date.date() == yesterday
    

    def scrape_and_analyze_news(self, max_days_back: int = 1, generate_images: bool = True) -> List[NewsArticle]:
        """Scrape news and analyze with LLM for good news detection"""
        
        all_articles = []
        sources = self.news_sources
        
        # print(f"🔍 Scraping news for: {country}")
        
        for source_url in sources:
            print(f"📡 Fetching from: {source_url}")
            articles = self.fetch_rss_feed(source_url)
            recent_count = 0
            for article_data in articles:
                try:
                    # Parse published date first
                    published_date = datetime.now()
                    if article_data["published"]:
                        try:
                            published_date = datetime(*article_data["published"][:6])
                        except (TypeError, ValueError):
                            published_date = datetime.now()

                    # Check if article is recent enough BEFORE analysis
                    if not self.is_recent_article(published_date, 1):
                        continue
                    # 1. Skip video articles
                    if "/av/" in article_data["link"] or "video" in article_data["link"]:
                        continue

                    # 2. Skip if title or url was used already
                    title_lc = article_data["title"].strip().lower()
                    url = article_data["link"]
                    if(len(self.previous_articles)>1):
                        if any(p["title"] == title_lc or p["url"] == url for p in self.previous_articles):
                            continue
                    
                    if(len(all_articles)>1):
                        if any(p.title == title_lc or p.url == url for p in all_articles):
                            continue


                    embedding = self.llm_analyzer.get_embedding(article_data["summary"])
                    to_pass = False
                    if(len(self.previous_articles)>1):
                        for i, article in enumerate(self.previous_articles):
                            if article["embedding"] == "":
                                article["embedding"] = self.llm_analyzer.get_embedding(article["summary"])
                                time.sleep(0.5)  # be nice to OpenAI rate limits
                            score = cosine_similarity(embedding, article["embedding"])          
                            if score >= SIMILARITY_THRESHOLD:   
                                print(f"Skipping duplicate article: {article_data['title']} (similar to previous article with score {score:.2f})")
                                to_pass = True
                                break
                        if to_pass:
                            continue

                    # print("Checked embeedings")
                    if(len(all_articles)>1):
                        for i, article in enumerate(all_articles):
                            # print(article)
                            if not article.embedding:
                                continue
                            score = cosine_similarity(embedding, article.embedding)          
                            if score >= SIMILARITY_THRESHOLD:   
                                print(f"Skipping duplicate article: {article_data['title']} (similar to previous article with score {score:.2f})")
                                to_pass = True
                                break
                        if to_pass:
                            continue
                    
                    # Analyze with LLM
                    print(f"🤖 Analyzing: {article_data['title'][:50]}...")
                    analysis = self.llm_analyzer.analyze_news_sentiment(
                        article_data["title"],
                        article_data["summary"],
                        article_data["content"]
                    )

                    if analysis["is_good_news"]:
                        recent_count += 1
                        # if recent_count > max_articles:
                        article = NewsArticle(
                            title=article_data['title'],
                            summary=article_data['summary'],
                            content=article_data['content'],
                            url=article_data['link'],
                            source=article_data['source'],
                            published=published_date,
                            category=analysis['category'],
                            sentiment_score=analysis['sentiment_score'],
                            is_good_news=True,
                            reasoning=analysis['reasoning'],
                            embedding=embedding
                        )
                        
                        all_articles.append(article)
                        print(f"✅ Good news found in {analysis['category']}! Score: {analysis['sentiment_score']:.2f}")
                    # else:
                    #     print(f"❌ Not good news: {analysis['reasoning']}")
                        
                except Exception as e:
                    print(f"Error analyzing article: {e}")
                
            time.sleep(1)  # Be respectful to servers and API limits
                
        # Sort by sentiment score
        all_articles.sort(key=lambda x: x.sentiment_score, reverse=True)
        return all_articles

    def present_news_with_personality(self, articles: List[NewsArticle], personality: str) -> List[Dict]:
        """Generate personality-based presentations with title + text"""
        presentations = []
        
        print(f"🎭 Generating {personality} presentations...")
        
        for i, article in enumerate(articles, 1):
            print(f"🎪 Creating presentation {i}/{len(articles)}...")
            presentation = self.llm_analyzer.generate_personality_response(article, personality)
            presentations.append(presentation)
            print(presentation)
            time.sleep(0.5)
        
        return presentations

    def present_news_without_personality(self, articles: List[NewsArticle], personality: str) -> List[Dict]:
        """Generate personality-based presentations with title + text"""
        presentations = []
        
        print(f"🎭 Generating {personality} presentations...")
        
        for i, article in enumerate(articles, 1):
            print(f"🎪 Creating presentation {i}/{len(articles)}...")
            presentation = self.llm_analyzer.generate_summary_response(article, personality)
            presentations.append(presentation)
            print(presentation)
            time.sleep(0.5)
        
        return presentations

def protect_bold_sections(text):
    """
    Replace **section** with <b>section</b>
    """
    return re.sub(r"\*\*(.*?)\*\*", r"<b> \1 </b>", text)

def restore_bold_sections(text):
    """
    Replace <b>section</b> with **section**
    """
    return re.sub(r"<b> (.*?) </b>", r"**\1**", text)
# -------------------------------
def translate_text_deepl(text, deepl_key, target_lang="FR"):
    auth_key = deepl_key
    if not auth_key:
        raise ValueError("DEEPL_API_KEY environment variable not set")
    
    translator = deepl.Translator(auth_key)
    protect = protect_bold_sections(text)
    result = translator.translate_text(protect, target_lang=target_lang)
    translated = restore_bold_sections(result.text)
    return translated


def translate_text_google(text, target_lang, source_lang="en"):

    # Protect formatting (e.g., bold)
    protect = protect_bold_sections(text)

    # Initialize translator
    translator = GoogleTranslator(source=source_lang, target=target_lang.lower())

    # Translate
    translated_text = translator.translate(protect)

    # Restore formatting
    translated_text = restore_bold_sections(translated_text)

    print("✅ Translation successful")

    return translated_text

    
def generate_daily_good_news(openai_api_key, use_dall_e, cf_api_token, cf_account_id, deepl_api_key, personality='darth_vader', max_articles=10, generate_images=True):
    scraper = GoodNewsScraper(llm_api_key=openai_api_key, use_dall_e=use_dall_e, cf_api_token=cf_api_token, cf_account_id=cf_account_id)
    articles = scraper.scrape_and_analyze_news(generate_images=generate_images)
    unique_articles = {}
    for article in articles:
        if article.url not in unique_articles:
            unique_articles[article.url] = article
            # break
    # Convert back to list
    articles = list(unique_articles.values())

    articles_by_category = defaultdict(list)
    for article in articles:
        category = getattr(article, "category", "Other") or "Other"
        articles_by_category[category].append(article)

    selected_articles = []
    for category, category_articles in articles_by_category.items():
        sorted_articles = sorted(category_articles, key=lambda x: x.sentiment_score, reverse=True)
        top_10 = sorted_articles[:max_articles]
        selected_articles.extend(top_10)

    # Now generate personality text only for the selected ones
    presentations = scraper.present_news_without_personality(selected_articles, personality)

    results_by_category = {
    "Health": [],
    "Environment": [],
    "Technology": [],
    "Human Rights": [],
    "Space": [],
    "Other": []
    }
    for article, presentation_data in zip(selected_articles, presentations):
        if(generate_images):
            image_prompt = scraper.llm_analyzer.generate_image_prompt(article)
            # print(image_prompt)
            article.image_prompt = image_prompt
            generated_image = scraper.llm_analyzer.generate_image_cf(image_prompt, article.title)
            article.image_url = generated_image

        if not article.image_url:
            fallback_images = [
                os.path.join('public/images', filename)
                for filename in sorted(os.listdir('public/images'))
                if filename.startswith("inspiration_") 
            ]
            article.image_url = random.choice(fallback_images)

        # french_title = translate_text_deepl(presentation_data["title"], deepl_api_key,target_lang="FR")
        # french_text = translate_text_deepl(presentation_data["text"],deepl_api_key, target_lang="FR")

        # # Translate to Spanish
        # spanish_title = translate_text_deepl(presentation_data["title"], deepl_api_key, target_lang="ES")
        # spanish_text = translate_text_deepl(presentation_data["text"], deepl_api_key, target_lang="ES")
        # Translate to French
        # french_title = translate_text_google(presentation_data["title"], target_lang="fr")
        # french_text = translate_text_google(presentation_data["text"], target_lang="fr")

        # # Translate to Spanish
        # spanish_title = translate_text_google(presentation_data["title"], target_lang="es")
        # spanish_text = translate_text_google(presentation_data["text"], target_lang="es")

        category = article.category
        results_by_category[category].append({
            "title": article.title,
            "summary": article.summary,
            "content": article.content,
            "embedding": article.embedding,
            "url": article.url,
            "source": article.source,
            "published": article.published.strftime("%Y-%m-%d"),
            "sentiment_score": article.sentiment_score,
            "reasoning": article.reasoning,
            "category": category,
            "personality_title": presentation_data["title"],
            "personality_presentation": presentation_data["text"],
            "personality_title_fr": presentation_data["title_fr"],
            "personality_presentation_fr": presentation_data["text_fr"],
            "personality_title_es": presentation_data["title_es"],
            "personality_presentation_es": presentation_data["text_es"],
            "image_url": article.image_url,
            "image_prompt": article.image_prompt
        })

    for category, articles in results_by_category.items():
        if not articles:
            continue
        # Limit to 10 articles per category
        articles = articles[:max_articles]
        summary_en = scraper.llm_analyzer.generate_category_summary(articles, category)
        summary_fr = translate_text_deepl(summary_en, deepl_api_key, target_lang="FR")
        summary_es = translate_text_deepl(summary_en, deepl_api_key, target_lang="ES")
        filename = f"public/data/{datetime.now().strftime('%Y-%m-%d')}_{category.lower().replace(' ', '_')}_test.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "personality": personality,
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "news_summary": summary_en,
                "news_summary_fr": summary_fr,
                "news_summary_es": summary_es,
                "articles": articles
            }, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Saved: {filename}")


    return results_by_category
