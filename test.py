import requests
from bs4 import BeautifulSoup

url = "https://www.lemonde.fr/en/culture/article/2025/07/14/at-paris-grand-palais-denmark-and-france-unite-through-tapestry-art_6743360_30.html"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html",
}

try:
    r = requests.get(url, headers=headers)
    print("Status code:", r.status_code)
    print("Content snippet:", r.text[:500])
except Exception as e:
    print("Error:", e)