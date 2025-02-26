import requests
from bs4 import BeautifulSoup

# URL of the website to analyze for SEO
url = "https://aws.amazon.com/vi/what-is/python/"

# Send a request to fetch the HTML data
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Get the title tag
    title = soup.title.string if soup.title else "No title tag found"
    
    # Get the meta description tag
    meta_description = soup.find("meta", attrs={"name": "description"})
    description_content = meta_description["content"] if meta_description else "No meta description found"
    
    # Get heading tags (H1, H2, H3)
    h1_tags = [h1.text.strip() for h1 in soup.find_all("h1")]
    h2_tags = [h2.text.strip() for h2 in soup.find_all("h2")]
    h3_tags = [h3.text.strip() for h3 in soup.find_all("h3")]
    
    # Get a list of links
    links = [a["href"] for a in soup.find_all("a", href=True)]

    # Print results
    print("🚀 SEO Analysis for Page:", url)
    print("\n📌 Title:", title)
    print("\n📌 Meta Description:", description_content)
    print("\n📌 H1 Tags:", h1_tags if h1_tags else "No H1 tags found")
    print("\n📌 H2 Tags:", h2_tags if h2_tags else "No H2 tags found")
    print("\n📌 H3 Tags:", h3_tags if h3_tags else "No H3 tags found")
    print("\n📌 Total Links:", len(links))
    print("📌 Sample Links:", links[:5])  # Display first 5 links
else:
    print("⚠️ Unable to access the website. Please check the URL!")
