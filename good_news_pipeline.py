import requests
import feedparser
import json
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

@dataclass
class NewsArticle:
    title: str
    content: str
    url: str
    source: str
    published: datetime
    country: str
    sentiment_score: float = 0.0
    is_good_news: bool = False
    reasoning: str = ""
    image_url: str = ""
    image_prompt: str = ""

class LLMAnalyzer:
    """Handle all LLM interactions for sentiment analysis, personality generation, and image prompts"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        
        # Fallback to local/alternative LLM if OpenAI not available
        self.use_openai = api_key is not None
        
    def analyze_news_sentiment(self, title: str, content: str) -> Dict:
        """Analyze if news is positive and get sentiment score using LLM"""
        
        prompt = f"""
        Analyze this news article and determine if it's "good news" that would make most people feel positive and hopeful. Give it a sentiment score between 0 and 1, where 1 means it is really uplifting and positive.

        TITLE: {title}
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

        Respond in JSON format:
        {{
            "is_good_news": true/false,
            "sentiment_score": 0.00-1.00,
            "reasoning": "Brief explanation of why this is or isn't good news",
            "key_positive_elements": ["list", "of", "positive", "aspects"],
            "emotional_impact": "uplifting/inspiring/heartwarming/hopeful/etc"
        }}
        """
        
        if self.use_openai and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing news sentiment and identifying positive, uplifting stories."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                
                result = json.loads(response.choices[0].message.content)
                return result
            except Exception as e:
                print(f"LLM Analysis Error: {e}")
                return self._fallback_analysis(title, content)
        else:
            return self._fallback_analysis(title, content)
    
    def _fallback_analysis(self, title: str, content: str) -> Dict:
        """Fallback analysis when LLM is not available"""
        text = f"{title} {content}".lower()
        
        positive_indicators = [
            'help', 'save', 'rescue', 'cure', 'breakthrough', 'success', 'achieve',
            'donate', 'charity', 'volunteer', 'inspire', 'hope', 'recover',
            'celebrate', 'win', 'award', 'hero', 'innovation', 'solution'
        ]
        
        negative_indicators = [
            'death', 'kill', 'murder', 'attack', 'crash', 'disaster', 'crisis',
            'scandal', 'arrest', 'guilty', 'fire', 'flood', 'war', 'violence'
        ]
        
        positive_count = sum(1 for word in positive_indicators if word in text)
        negative_count = sum(1 for word in negative_indicators if word in text)
        
        is_good = positive_count > negative_count and positive_count > 0
        score = min(positive_count * 0.2, 1.0) if is_good else 0.0
        
        return {
            "is_good_news": is_good,
            "sentiment_score": score,
            "reasoning": f"Found {positive_count} positive and {negative_count} negative indicators",
            "key_positive_elements": ["fallback analysis"],
            "emotional_impact": "positive" if is_good else "neutral"
        }
    
    def generate_image_prompt(self, article: NewsArticle) -> str:
        """Generate DALL-E image prompt based on article content"""
        
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
        
        if self.use_openai and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at creating detailed, positive image prompts for DALL-E based on good news stories."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=150
                )
                
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Image Prompt Generation Error: {e}")
                return self._fallback_image_prompt(article)
        else:
            return self._fallback_image_prompt(article)
    
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
    
    def generate_image(self, prompt: str, article_title: str) -> Optional[str]:
        """Generate image using DALL-E and save locally"""
        if not self.use_openai or not self.client:
            print("OpenAI API not available for image generation")
            return None
        
        try:
            print(f"🎨 Generating image with prompt: {prompt[:100]}...")
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
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
                os.makedirs('static/generated_images', exist_ok=True)
                
                filename = f"static/generated_images/news_image_{safe_title}_{timestamp}.png"
                
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
    
    def generate_personality_response(self, article: NewsArticle, personality: str) -> str:
        """Generate personality-based news presentation using LLM"""
        
        personality_prompts = {
            'darth_vader': {
                'character': "Darth Vader from Star Wars",
                'traits': "Dramatic, ominous, speaks in a deep commanding voice, uses Star Wars terminology, but finds unexpected hope in good news. Often surprised by acts of kindness.",
                'catchphrases': "Uses phrases like 'Most impressive', 'I find your lack of faith disturbing', 'The Force is strong', 'You underestimate the power'"
            },
            'gary_lineker': {
                'character': "Gary Lineker, enthusiastic British football commentator and TV presenter",
                'traits': "Energetic, uses football metaphors, genuinely excited about good news, often relates stories to teamwork and fair play",
                'catchphrases': "Uses phrases like 'What a cracker!', 'Sensational stuff!', 'Pure class!', 'That's why we love this beautiful game called life!'"
            },
            'drunk_philosopher': {
                'character': "A slightly intoxicated philosopher at a pub",
                'traits': "Slurred speech (indicated by *hiccup*, *burp*), surprisingly profound insights, emotional, finds deep meaning in simple good news",
                'catchphrases': "Uses *hiccup*, *sway*, *raises glass*, gets emotional about humanity, talks about the meaning of life"
            },
            'shakespeare': {
                'character': "William Shakespeare, the famous playwright and poet",
                'traits': "Eloquent, uses iambic pentameter occasionally, finds poetic beauty in modern news, uses Elizabethan English mixed with understanding of modern world",
                'catchphrases': "Uses 'Hark!', 'Pray tell', 'What noble tale', 'All's well that ends well', 'What a piece of work is man'"
            },
            'gordon_ramsay': {
                'character': "Gordon Ramsay, passionate celebrity chef",
                'traits': "Intense, passionate, uses cooking metaphors, occasionally swears (but keeps it mild), gets genuinely emotional about good news",
                'catchphrases': "Uses 'Bloody hell!', 'That's beautiful!', 'Absolutely stunning!', references cooking and kitchen"
            }
        }
        
        if personality not in personality_prompts:
            personality = 'darth_vader'
        
        char_info = personality_prompts[personality]
        
        prompt = f"""
        You are {char_info['character']}. 

        CHARACTER TRAITS: {char_info['traits']}
        TYPICAL PHRASES: {char_info['catchphrases']}

        Present this good news story in your characteristic style. Make it entertaining and engaging while staying true to your character. Keep it to 3-4 paragraphs maximum.

        NEWS TITLE: {article.title}
        NEWS CONTENT: {article.content}
        SOURCE: {article.source}
        PUBLISHED: {article.published.strftime('%Y-%m-%d')}
        POSITIVITY ELEMENTS: {article.reasoning}

        Format your response as the character would speak, including any characteristic verbal tics, mannerisms, or speech patterns. Make it fun and engaging!
        """
        
        if self.use_openai and self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": f"You are {char_info['character']} presenting good news. Stay in character completely."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.8,
                    max_tokens=400
                )
                
                return response.choices[0].message.content
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

class GoodNewsScraper:
    def __init__(self, llm_api_key: str = None):
        self.llm_analyzer = LLMAnalyzer(api_key=llm_api_key)
        
        self.news_sources = {
            'global': [
                'https://feeds.bbci.co.uk/news/rss.xml',
                'https://rss.cnn.com/rss/edition.rss',
                'https://feeds.reuters.com/reuters/topNews',
                'https://feeds.npr.org/1001/rss.xml',
                'http://feeds.abcnews.com/abcnews/topstories',
            ],
            'us': [
                'https://rss.cnn.com/rss/cnn_us.rss',
                'https://feeds.npr.org/1003/rss.xml',
                'http://feeds.abcnews.com/abcnews/usheadlines',
            ],
            'uk': [
                'https://feeds.bbci.co.uk/news/uk/rss.xml',
                'https://www.theguardian.com/uk/rss',
            ],
            'canada': [
                'https://feeds.cbc.ca/rss/canada',
            ],
            'australia': [
                'https://feeds.abc.net.au/news/australia',
            ],
            'tech': [
                'https://feeds.feedburner.com/TechCrunch',
                'https://www.wired.com/feed/rss',
            ],
            'science': [
                'https://www.sciencedaily.com/rss/top.xml',
                'https://www.nature.com/nature.rss',
            ]
        }

    def fetch_rss_feed(self, url: str) -> List[Dict]:
        """Fetch and parse RSS feed"""
        try:
            response = requests.get(url, timeout=10)
            feed = feedparser.parse(response.content)
            
            articles = []
            for entry in feed.entries[:15]:  # Limit to recent articles
                articles.append({
                    'title': entry.get('title', ''),
                    'summary': entry.get('summary', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published_parsed', None),
                    'source': feed.feed.get('title', url)
                })
            return articles
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return []

    def is_recent_article(self, published_date: datetime, days_back: int = 2) -> bool:
        """Check if article was published within the specified number of days"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        return published_date >= cutoff_date
    
    def scrape_and_analyze_news(self, country: str = 'global', max_articles: int = 10, max_days_back: int = 3, generate_images: bool = True) -> List[NewsArticle]:
        """Scrape news and analyze with LLM for good news detection"""
        if country not in self.news_sources:
            print(f"Country '{country}' not supported. Available: {list(self.news_sources.keys())}")
            return []
        
        all_articles = []
        sources = self.news_sources[country]
        
        print(f"🔍 Scraping news for: {country}")
        
        for source_url in sources:
            print(f"📡 Fetching from: {source_url}")
            articles = self.fetch_rss_feed(source_url)
            recent_count = 0
            for article_data in articles:
                try:
                    # Analyze with LLM
                    print(f"🤖 Analyzing: {article_data['title'][:50]}...")
                    analysis = self.llm_analyzer.analyze_news_sentiment(
                        article_data['title'], 
                        article_data['summary']
                    )
                    
                    if analysis['is_good_news']:
                        published_date = datetime.now()
                        if article_data['published']:
                            try:
                                published_date = datetime(*article_data['published'][:6])
                            except (TypeError, ValueError):
                                # If parsing fails, assume it's recent
                                published_date = datetime.now()
                        
                        # Check if article is recent enough
                        if not self.is_recent_article(published_date, max_days_back):
                            print(f"⏰ Skipping old article: {article_data['title'][:50]}... (Published: {published_date.strftime('%Y-%m-%d')})")
                            continue
                    
                        recent_count += 1
                        article = NewsArticle(
                            title=article_data['title'],
                            content=article_data['summary'],
                            url=article_data['link'],
                            source=article_data['source'],
                            published=published_date,
                            country=country,
                            sentiment_score=analysis['sentiment_score'],
                            is_good_news=True,
                            reasoning=analysis['reasoning']
                        )
                        
                        # # Generate image if requested
                        # if generate_images:
                        #     image_prompt = self.llm_analyzer.generate_image_prompt(article)
                        #     article.image_prompt = image_prompt
                        #     article.image_url = self.llm_analyzer.generate_image(image_prompt, article.title)
                        
                        all_articles.append(article)
                        print(f"✅ Good news found! Score: {analysis['sentiment_score']:.2f}")
                    else:
                        print(f"❌ Not good news: {analysis['reasoning']}")
                        
                except Exception as e:
                    print(f"Error analyzing article: {e}")
            
            time.sleep(1)  # Be respectful to servers and API limits
        
        # Sort by sentiment score
        all_articles.sort(key=lambda x: x.sentiment_score, reverse=True)
        return all_articles[:max_articles]


    def present_news_with_personality(self, articles: List[NewsArticle], personality: str) -> List[str]:
        """Generate personality-based presentations for all articles"""
        presentations = []
        
        print(f"🎭 Generating {personality} presentations...")
        
        for i, article in enumerate(articles, 1):
            print(f"🎪 Creating presentation {i}/{len(articles)}...")
            presentation = self.llm_analyzer.generate_personality_response(article, personality)
            presentations.append(presentation)
            time.sleep(0.5)  # Rate limiting
        
        return presentations


def generate_daily_good_news(api_key, country='global', personality='darth_vader', max_articles=3, generate_images=True):
    scraper = GoodNewsScraper(llm_api_key=api_key)
    articles = scraper.scrape_and_analyze_news(country, max_articles, generate_images=generate_images)
    presentations = scraper.present_news_with_personality(articles, personality)

    results = {
        'country': country,
        'personality': personality,
        'timestamp': datetime.now().isoformat(),
        'articles': []
    }
    for article, presentation in zip(articles, presentations):
        if(generate_images):
            image_prompt = scraper.llm_analyzer.generate_image_prompt(article)
            article.image_prompt = image_prompt
            article.image_url = scraper.llm_analyzer.generate_image(image_prompt, article.title)
        results['articles'].append({
            'title': article.title,
            'content': article.content,
            'url': article.url,
            'source': article.source,
            'published': article.published.strftime("%Y-%m-%d"),
            'sentiment_score': article.sentiment_score,
            'reasoning': article.reasoning,
            'personality_presentation': presentation,
            'image_url': article.image_url,
            'image_prompt': article.image_prompt
        })
    return results
