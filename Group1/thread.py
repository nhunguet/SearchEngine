import requests
from bs4 import BeautifulSoup

def crawl(url):
    print(f"Crawling: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').text if soup.find('title') else 'No title'
        print(f"Title of {url}: {title}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to crawl {url}: {e}")

urls = [
    'https://www.youtube.com/'
]

threads = []
for url in urls:
    thread = crawl(url)
    threads.append(thread)
