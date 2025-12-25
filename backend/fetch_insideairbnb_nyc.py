# backend/fetch_insideairbnb_nyc.py

import requests
import pandas as pd
from io import BytesIO
import gzip
import os
from bs4 import BeautifulSoup

BASE_URL = "http://insideairbnb.com/get-the-data.html"


def get_latest_nyc_folder():
    """Scrape InsideAirbnb to find latest NYC dataset folder."""
    html = requests.get(BASE_URL).text
    soup = BeautifulSoup(html, "html.parser")

    links = soup.find_all("a")
    nyc_links = [l.get("href") for l in links if l.get("href") and "new-york-city" in l.get("href")]

    if not nyc_links:
        raise RuntimeError("No NYC datasets found on InsideAirbnb!")

    # Latest dataset = last item
    return nyc_links[-1]


def download_and_extract(url, output_path):
    """Download .csv.gz and save decompressed CSV."""
    print(f"Downloading: {url}")

    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to download dataset from {url}")

    gz_data = BytesIO(r.content)
    with gzip.open(gz_data, 'rb') as gz:
        csv_data = gz.read()

    with open(output_path, "wb") as f:
        f.write(csv_data)

    print(f"Saved: {output_path}")


def fetch_latest_nyc_data(save_dir="../data"):
    """Fetch the latest NYC datasets and save locally."""
    os.makedirs(save_dir, exist_ok=True)

    latest_folder = get_latest_nyc_folder()

    datasets = {
        "listings.csv.gz": "listings.csv",
        "calendar.csv.gz": "calendar.csv",
        "reviews.csv.gz": "reviews.csv"
    }

    for gz, out in datasets.items():
        file_url = f"{latest_folder}{gz}"
        output_path = os.path.join(save_dir, out)
        download_and_extract(file_url, output_path)

    print("All NYC datasets fetched successfully!")
    return True


if __name__ == "__main__":
    fetch_latest_nyc_data()
