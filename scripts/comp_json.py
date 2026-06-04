import json
import os

def load_articles(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("articles", [])

def compare_presentations(test_file, prod_file):
    test_articles = load_articles(test_file)
    prod_articles = load_articles(prod_file)

    # Map by URL
    test_map = {a.get("url"): a for a in test_articles if a.get("url")}
    prod_map = {a.get("url"): a for a in prod_articles if a.get("url")}

    for url, test_article in test_map.items():
        if url in prod_map:
            prod_article = prod_map[url]
            print("="*80)
            print(f"URL: {url}")
            print("\nTEST:")
            print("Title:", test_article.get("title", ""))
            print("Presentation:", test_article.get("personality_presentation", ""))
            print("\nPROD:")
            print("Title:", prod_article.get("title", ""))
            print("Presentation:", prod_article.get("personality_presentation", ""))
            print("="*80 + "\n")

if __name__ == "__main__":
    # Example for space category
    test_file = "public/data/2025-08-10_health_test.json"
    prod_file = "public/data/2025-08-10_health.json"
    compare_presentations(test_file, prod_file)
