from good_news_pipeline import GoodNewsScraper  # replace with actual module name
import datetime
def test_rss_parsing():
    scraper = GoodNewsScraper()
    
    # Choose one URL to test (use one with consistent structure like BBC)
    test_url = scraper.news_sources
    
    for url in test_url[:1]:
        print(f"Fetching articles from: {url}")
        articles = scraper.fetch_rss_feed(url)

        print(f"\n✅ {len(articles)} articles fetched from {url}\n")
        
        for i, article in enumerate(articles):
            print(f"{i+1}. {article.get('title', '[No Title]')}")
            print(f" SUMMARY  {article.get('summary', '[No Summary]')}")
            print(f"Content len: {len(article.get('content', ''))}")
            # print(f"Content: ", article.get('content', '[No Content]')[:])
            print(f"Date: {article.get('published_parsed', '[No Date]')}")
            print(f"Date: {article.get('published', '[No Date]')}")
            if article.get('published'):
                published_date = datetime.datetime(*article.get('published')[:6])
                print(f"Formatted Date: {published_date.strftime('%Y-%m-%d %H:%M:%S')}")
            elif article.get('published_parsed'):
                print('PARSED')

            # print(f" DESCRIPTION  {article.get('description', '[No Description]')}")
            # print(f" CONTENT  {article.get('content', '[No Content]')[:50]}")


        # break
if __name__ == "__main__":
    test_rss_parsing()
