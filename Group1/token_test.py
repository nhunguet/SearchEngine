import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt_tab')

# Define a function to fetch and clean text from a Wikipedia page
def fetch_wikipedia_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for script in soup(["script", "style"]):
        script.decompose()
        
    text = soup.get_text()
    return text

# Define a function to tokenize and filter text
def tokenize_and_filter(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Filter tokens that are made up of only English alphabetic characters
    filtered_tokens = [token for token in tokens if token.isalpha()]
    return filtered_tokens

# URLs of Wikipedia pages you want to download
urls = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Deep_learning"
]

# Fetch and process each page
all_tokens = []
for url in urls:
    text = fetch_wikipedia_page(url)
    tokens = tokenize_and_filter(text)
    all_tokens.extend(tokens)

print(all_tokens[:100])  # Print first 100 tokens to check
