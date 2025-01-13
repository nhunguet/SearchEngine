import threading
import requests

def check_last_modified(url):
    print(f"Checking last modified time for: {url}")
    try:
        response = requests.head(url)
        response.raise_for_status()  # Check for request errors
        last_modified = response.headers.get('Last-Modified', 'No Last-Modified header found')
        print(f"Last-Modified for {url}: {last_modified}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to check {url}: {e}")

urls = [
    'https://en.wikipedia.org/wiki/Cat'
]

threads = []
for url in urls:
    thread = check_last_modified(url)
    threads.append(thread)