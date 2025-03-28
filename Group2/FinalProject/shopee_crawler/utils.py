import requests
import pandas as pd
import os, time, random, re, csv
from urllib.parse import urlparse
import json
from playwright.async_api import async_playwright
import asyncio


def load_urls(url_file):
    try:
        df = pd.read_csv(url_file)
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "url", "status"])


def update_url_status(url, status, url_file):
    df = load_urls(url_file)
    df.loc[df["url"] == url, "status"] = status
    df.to_csv(url_file, index=False)


def reset_url_status(url_file):
    df = load_urls(url_file)
    df["status"] = "not scraped"
    df.to_csv(url_file, index=False)
    try:
        os.remove('reviews.csv')
    except FileNotFoundError:
        print("reviews.csv not found, cannot delete.")


def extract_shopee_ids(url):
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path
        match = re.search(r"i\.(\d+)\.(\d+)", path)

        if match:
            shop_id = int(match.group(1))
            item_id = int(match.group(2))
            return shop_id, item_id
        
    except Exception as e:
        print(f"Error parsing URL: {e}")
        return None
    

def preprocess_comment(comment):
    # Remove multiple spaces
    comment = re.sub(r"\s+", " ", comment).strip()
    # Remove quotation marks
    comment = comment.replace('"', "")
    # Convert to lowercase
    return comment.lower()


def initialize_csv(file_path, columns):
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        pd.DataFrame(columns=columns).to_csv(file_path, index=False, encoding="utf-8")


async def get_shopee_reviews(shop_id, item_id):
    review_list = []
    limit = 15  # Shopee allows a max of 50 per request
    target_count = {"1": 15, "2": 15, "3": 15, "4": 15, "5": 15}
    collected_reviews = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set to False to see browser
        context = await browser.new_context()

        with open("shopee_cookies.json", "r") as f:
            cookies = json.load(f)

        await context.add_cookies(cookies)
        page = await context.new_page()

        try:
            await page.goto("https://shopee.vn")
            await asyncio.sleep(10)

            for rating in collected_reviews.keys():
                offset = 0  # Reset offset

                while collected_reviews[rating] < target_count[rating]:
                    api_url = f"https://shopee.vn/api/v2/item/get_ratings?itemid={item_id}&shopid={shop_id}&limit={limit}&offset={offset}&type={rating}"
                    
                    # Fetch data using the browser context
                    response = await page.evaluate(f"() => fetch('{api_url}').then(res => res.json())")
                    
                    data = response.get("data", {}).get("ratings", [])
                    if not data:
                        break

                    for r in data:
                        if collected_reviews[rating] >= target_count[rating]:
                            break 
                        
                        try:
                            comment = r.get("comment", "")      # If there is comment for the rating
                        except:
                            print(f'no comment for rating {rating}')
                            break

                        if comment:        # If there is comment AND comment is not empty
                            review_list.append({
                                "comment": preprocess_comment(comment),
                                "label": None
                            })
                            collected_reviews[rating] += 1

                    offset += limit  # Move to the next batch
                    await asyncio.sleep(random.uniform(2, 5))
                
        except asyncio.CancelledError:
            print("Scraping cancelled. Closing browser.")
            await browser.close()
            raise  # Re-raise the exception to be handled by `scrape_reviews()`

        finally:
            await browser.close()

    if review_list:
        pd.DataFrame(review_list).to_csv('reviews.csv', mode='a', header=False, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)


async def scrape_reviews(url_file):
    df = load_urls(url_file)
    df_unprocessed = df[df["status"] != "scraped"]
    if df_unprocessed.empty:
        print("No URLs left to scrape.")
        return True

    initialize_csv('reviews.csv', ["comment", "label"])

    for _, row in df_unprocessed.iterrows():
        name, url = row["name"], row["url"]

        print(f"Scraping {name}...")

        try:
            shop_id, item_id = extract_shopee_ids(url)
            await get_shopee_reviews(shop_id, item_id)
            update_url_status(url, "scraped", url_file)

        except asyncio.CancelledError:
            print("Scraping was cancelled. Cleaning up...")
            return False  # Gracefully exit if cancelled

        except Exception as e:
            print(f"Error scraping {url}: {e}")
            update_url_status(url, "failed", url_file)

        await asyncio.sleep(2)

    print("One epoch completed.")
    return False #urls remain to be scraped
