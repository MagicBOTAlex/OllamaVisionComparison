import base64
import os
import re
import httpx
import asyncio
import pandas as pd
import time
from collections import defaultdict
import random

resultCsvPath = "./results.csv"
comparingModel = "gemma3:12b"
predCol = "gemma_pred"

def get_all_images():
    images = []
    for root, _, files in os.walk("./data/UTKFaceData/"):
        for file in files:
            # only include .jpg files
            if file.lower().endswith(".jpg"):
                images.append(os.path.join(root, file))
    return images

all_images = get_all_images()

def get_shuffled_filtered():
# Group images by race (parsed from filenames)
    images_by_race = defaultdict(list)
    for img_path in all_images:
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        if len(parts) >= 4:
            race = parts[2]
            images_by_race[race].append(img_path)

# Randomly sample up to 1500 images per race
    sampled_images = []
    for race, imgs in images_by_race.items():
        count = min(500, len(imgs))
        sampled = random.sample(imgs, count)
        sampled_images.extend(sampled)

# Optionally shuffle the final list
    random.shuffle(sampled_images)
    return sampled_images

genderPrompt = """
which gender does the person in the image have?
start by describing the details of the image, then facial features.
when ready, then place the final answer of gender in bold text. (Either male or female)
"""


# load or initialize the dataframe
if os.path.exists(resultCsvPath):
    df = pd.read_csv(resultCsvPath)
else:
    df = pd.DataFrame(columns=["image", "ethnicity", "actual_gender", "llama_pred", "gemma_pred"])

    # Ensure every image has a row
    for path in get_shuffled_filtered():
        if path not in df["image"].values:
            df.loc[len(df)] = [path, None, None, None, None]

    df.to_csv(resultCsvPath, index=False)

async def generate_image_text(client: httpx.AsyncClient, base_url: str, auth: tuple, model: str, image_path: str, prompt: str) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [img_b64]
    }
    resp = await client.post(f"{base_url}/api/generate", json=payload, timeout=20, auth=auth)
    resp.raise_for_status()
    return resp.json().get("response", "")

async def predict_missing():
    # start timing and counter for throughput
    start_time = time.monotonic()
    processed_count = 0

    # Build queue of indices that still need a prediction
    queue = asyncio.Queue()
    for idx, row in df.iterrows():
        if pd.isna(row[predCol]) or row[predCol] == "":
            queue.put_nowait((idx, row["image"]))

    if queue.empty():
        print("All images already have predictions.")
        return

    # Lock to serialize DataFrame + CSV writes
    lock = asyncio.Lock()

    async def worker(base_url: str, auth: tuple):
        nonlocal processed_count
        client = httpx.AsyncClient()
        try:
            while True:
                try:
                    idx, image_path = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                print(f"[{base_url}] Predicting for: {image_path}")
                try:
                    text = await generate_image_text(client, base_url, auth, comparingModel, image_path, genderPrompt)
                    # extract last non-empty line
                    lines = [l for l in text.strip().splitlines() if l.strip()]
                    last = lines[-1] if lines else ""
                    m = re.search(r"\b(male|female)\b", last, re.IGNORECASE)
                    gender = m.group(1).lower() if m else "unknown"

                    # parse ethnicity and actual_gender from filename
                    parts = os.path.basename(image_path).split('_')
                    ethnicity = parts[2]
                    actual_gender = parts[1]

                    # update dataframe safely
                    async with lock:
                        df.at[idx, predCol] = gender
                        df.at[idx, "ethnicity"] = ethnicity
                        df.at[idx, "actual_gender"] = actual_gender
                        df.to_csv(resultCsvPath, index=False)

                        # bump counter and print throughput
                        processed_count += 1
                        elapsed = time.monotonic() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0.0
                        sec_per_pred = elapsed / processed_count if processed_count > 0 else 0.0

                        print(
                            f"Saved: {gender}  |  "
                            f"Total: {processed_count}  |  "
                            f"{rate:.2f} pred/sec  |  "
                            f"{sec_per_pred:.2f} sec/pred\n"
                        )
                except Exception as e:
                    print(f"Error on {image_path}: {e}")

                queue.task_done()
        finally:
            await client.aclose()

    # two endpoints in parallel
    endpoints = [
        "http://localhost:11434",
        "http://192.168.50.58:11434",
        "https://ollama.spoodythe.one",
        "https://ollama.deprived.dev"
    ]
    auth = [
            None,
    None,
    ("zhen", "query-erratic-crushable-ivy-curler-unjustly-marbled-exhume"),
    ("zhen", "query-erratic-crushable-ivy-curler-unjustly-marbled-exhume")
            ]
    tasks = [asyncio.create_task(worker(endpoints[idx], auth[idx])) for idx in range(len(endpoints))]

    # wait until queue is fully processed
    await queue.join()
    # cancel workers (they’ll exit after queue is empty)
    for t in tasks:
        t.cancel()

if __name__ == "__main__":
    asyncio.run(predict_missing())

