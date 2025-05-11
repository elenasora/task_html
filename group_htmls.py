# -*- coding: utf-8 -*-
"""
Created on Wed May  7 18:45:31 2025

@author: Elena
"""

"""
Cum am gandit sa rezolv taskul: (detalii in observatii.txt)
1. Am o structura de foldere, fiecare folder contine mai multe fisiere HTML (se numesc tier1, tier2, etc.).
2. Pentru fiecare fisier HTML, am folosit Playwright pentru a-l deschide in browser, a face un screenshot si a extrage textul din pagina.
3. Am calculat un "hash" al imaginii pentru a compara vizual acele pagini si am folosit un model pre-antrenat pentru a extrage embedding-uri din textul paginii.
4. Am folosit clustering pentru a grupa paginile care sunt similare din punct de vedere vizual si al textului.
5. La final, am salvat grupurile formate intr-un fisier JSON pentru a putea analiza rezultatele.

"""

import os
import json
import asyncio
from pathlib import Path
from PIL import Image
import imagehash
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from playwright.async_api import async_playwright

# Incarc modelul pre-antrenat de la SentenceTransformers pentru a obtine embedding-uri pentru text
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configurez caile pentru directorul principal si pentru directoarele de screenshoturi
TIERS_PARENT_DIR = "clones"  # directorul principal cu subdirectoare tier
SCREENSHOTS_DIR = "screenshots"  # directorul unde vor fi salvate screenshoturile
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)  # creez directorul pentru screenshoturi daca nu exista

# Functia pentru a deschide fisierul HTML, a face un screenshot si a extrage textul
async def render_and_extract(file_path):
    # Folosesc Playwright pentru a deschide pagina HTML si a captura continutul
    async with async_playwright() as p:
        browser = await p.chromium.launch()  # Lansez browserul Chromium
        page = await browser.new_page()  # Creez o pagina noua
        url = Path(file_path).absolute().as_uri()  # Convertesc calea fisierului intr-un URL
        try:
            await page.goto(url, timeout=60000)  # Incerc sa incarc pagina (timeout de 1 minut)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")  # Daca nu se poate incarca, afisez eroarea
            return None, ""  # Returnez None pentru screenshot si text gol

        await page.set_viewport_size({"width": 1280, "height": 800})  # Setez dimensiunea ferestrei pentru captura
        screenshot_path = os.path.join(SCREENSHOTS_DIR, Path(file_path).stem + ".png")  # Generez calea pentru screenshot
        await page.screenshot(path=screenshot_path, full_page=True)  # Fac captura de ecran
        text = await page.inner_text("body")  # Extrag textul din corpul paginii
        await browser.close()  # Inchid browserul
        return screenshot_path, text  # Returnez calea screenshot-ului si textul

# Functia pentru a calcula hash-ul perceptual al imaginii
def compute_image_hash(image_path):
    image = Image.open(image_path).convert("L")  # Deschid imaginea si o convertesc la tonuri de gri
    return imagehash.phash(image)  # Calculez hash-ul perceptual al imaginii

# Functia pentru a grupa documentele pe baza hash-urilor imaginii si a embedding-urilor textului
def cluster_items(phashes, text_embeddings, files, threshold=0.4):
    # Calculez distantele intre hash-urile imaginilor
    image_distances = np.array([
        [phashes[i] - phashes[j] for j in range(len(files))]
        for i in range(len(files))
    ]) / 64.0  # Normalizez distantele hash-urilor (divizez la 64 pentru a le face mai usor de interpretat)

    # Calculez distantele cosinus intre embedding-urile textului
    text_distances = cosine_distances(text_embeddings)
    
    # Calculez distanta totala intre imagini si texte (80% imagine, 20% text)
    total_dist = 0.6 * image_distances + 0.4 * text_distances

    # Realizez clustering folosind distantele pre-computate
    clustering = AgglomerativeClustering(
        metric='precomputed',  # Folosesc distantele pre-computate
        linkage='average',  # Folosesc metoda de legatura medie
        distance_threshold=threshold,  # Pragul pentru separarea clusterelor
        n_clusters=None  # Nu specific un numar fix de clustere
    )
    labels = clustering.fit_predict(total_dist)  # Aplic algoritmul de clustering

    # Creez grupuri pe baza etichetelor de clustering
    groups = {}
    for idx, label in enumerate(labels):
        groups.setdefault(label, []).append(files[idx])  # Grupez fisierele in functie de eticheta obtinuta

    return list(groups.values())  # Returnez grupurile formate

# Functia care proceseaza fisierele HTML dintr-un tier (subdirector)
async def process_tier(tier_dir):
    # Caut toate fisierele HTML din directorul dat
    html_files = [str(Path(tier_dir) / f) for f in os.listdir(tier_dir) if f.endswith(".html")]
    print(f"Found {len(html_files)} HTML files in {tier_dir}")  # Afisez numarul de fisiere HTML gasite

    phashes = []  # Lista pentru hash-urile imaginilor
    embeddings = []  # Lista pentru embedding-urile textului
    processed_files = []  # Lista pentru fisierele procesate

    # Procesez fiecare fisier HTML
    for file in html_files:
        try:
            result = await render_and_extract(file)  # Extrag screenshot-ul si textul
            if result is None:
                continue  # Daca nu s-a putut extrage, trec la urmatorul fisier
            screenshot_path, text = result
            ph = compute_image_hash(screenshot_path)  # Calculez hash-ul pentru imagine
            emb = model.encode(text)  # Obtin embedding pentru text
            phashes.append(ph)  # Adaug hash-ul in lista
            embeddings.append(emb)  # Adaug embedding-ul in lista
            processed_files.append(Path(file).name)  # Adaug numele fisierului in lista de fisiere procesate
        except Exception as e:
            print(f"Error processing {file}: {e}")  # Daca apare vreo eroare, o afisez

    # Aplic clustering pe fisierele procesate
    groups = cluster_items(phashes, embeddings, processed_files)
    print(f"\nGrouped Results for {Path(tier_dir).name}:")  # Afisez rezultatele gruparii pentru fiecare tier
    for group in groups:
        print(group)  # Afisez fiecare grup

    return Path(tier_dir).name, groups  # Returnez numele tier-ului si grupurile formate

# Functia principala care proceseaza toate tier-ele si salveaza rezultatele intr-un fisier JSON
async def main():
    tiers = [d for d in os.listdir(TIERS_PARENT_DIR) if d.startswith("tier")]  # Obtin toate subdirectoarele care sunt tier
    all_results = {}  # Dictionar unde voi salva rezultatele

    # Procesez fiecare tier
    for tier in tiers:
        tier_path = os.path.join(TIERS_PARENT_DIR, tier)  # Caut calea completa a tier-ului
        tier_name, groups = await process_tier(tier_path)  # Procesez tier-ul
        all_results[tier_name] = groups  # Salvez grupurile în dictionar

    # Salvez toate rezultatele intr-un JSON
    with open("all_grouped_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)  # Scriu rezultatele in JSON


if __name__ == "__main__":
    asyncio.run(main())
