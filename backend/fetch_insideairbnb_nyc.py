\
#!/usr/bin/env python3
\"\"\"fetch_insideairbnb_nyc.py
Fetch latest InsideAirbnb NYC dataset folder by scraping the 'Get the data' page
and download listings.csv, calendar.csv, reviews.csv (if available).
Saves into ../data/
\"\"\"
import os, requests, sys, time
from bs4 import BeautifulSoup

BASE_PAGE = "https://insideairbnb.com/get-the-data.html"
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def find_nyc_folder():
    resp = requests.get(BASE_PAGE, timeout=20)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = [a.get("href") for a in soup.find_all("a", href=True)]
    # look for links that include 'new-york' or 'new-york-city'
    candidates = [l for l in links if l and ("new-york" in l.lower())]
    # prefer absolute insideairbnb links
    for c in candidates:
        if c.startswith("http") and "insideairbnb.com" in c.lower():
            return c.rstrip("/")
    if candidates:
        return candidates[0].rstrip("/")
    raise RuntimeError("Could not locate NYC folder on InsideAirbnb page. Please check manually.")

def download(url, out_path):
    print(f"Downloading {url} -> {out_path}")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return out_path

def main():
    print("Finding latest NYC folder on InsideAirbnb...")
    folder = find_nyc_folder()
    print("Folder found:", folder)
    files = ["listings.csv", "calendar.csv", "reviews.csv"]
    for fname in files:
        url = f"{folder}/{fname}"
        out = os.path.join(DATA_DIR, fname)
        try:
            download(url, out)
        except Exception as e:
            print(f"Warning: could not download {fname}: {e}")
    print("Done. Files saved to", DATA_DIR)
    print("If a file is missing, you can manually download it from:", folder)

if __name__ == '__main__':
    main()
