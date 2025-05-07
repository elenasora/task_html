# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:45:31 2025

@author: Elena
"""

# group_htmls.py

import os
import asyncio
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from playwright.async_api import async_playwright

model = SentenceTransformer("all-MiniLM-L6-v2")

# Config
TIERS_DIR = "clones/tier1"  # modifică pentru tier2, tier3 etc.
SCREENSHOTS_DIR = "screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

async def render_and_extract(file_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        url = Path(file_path).absolute().as_uri()
        try:
            await page.goto(url, timeout=60000)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
            return None, ""

        await page.set_viewport_size({"width": 1280, "height": 800})
        screenshot_path = os.path.join(SCREENSHOTS_DIR, Path(file_path).stem + ".png")
        await page.screenshot(path=screenshot_path, full_page=True)
        content = await page.content()
        text = await page.inner_text("body")
        await browser.close()
        return screenshot_path, text

def compute_image_hash(image_path):
    image = Image.open(image_path).convert("L")
    return imagehash.phash(image)

def cluster_items(phashes, text_embeddings, files, threshold=0.4):
    # Normalize
    image_distances = np.array([
        [phashes[i] - phashes[j] for j in range(len(files))]
        for i in range(len(files))
    ]) / 64.0  # max hash distance

    text_distances = cosine_distances(text_embeddings)

    total_dist = 0.6 * image_distances + 0.4 * text_distances

    clustering = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=threshold,
        n_clusters=None
    )
    labels = clustering.fit_predict(total_dist)

    groups = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(files[idx])
    return list(groups.values())

async def main():
    html_files = [str(Path(TIERS_DIR) / f) for f in os.listdir(TIERS_DIR) if f.endswith(".html")]

    print(f"Found {len(html_files)} HTML files in {TIERS_DIR}")

    phashes = []
    embeddings = []
    processed_files = []

    for file in html_files:
        try:
            result = await render_and_extract(file)
            if result is None:
                continue
            screenshot_path, text = result
            ph = compute_image_hash(screenshot_path)
            emb = model.encode(text)
            phashes.append(ph)
            embeddings.append(emb)
            processed_files.append(Path(file).name)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    groups = cluster_items(phashes, embeddings, processed_files)
    print("\nGrouped Results:")
    for group in groups:
        print(group)

if __name__ == "__main__":
    asyncio.run(main())
