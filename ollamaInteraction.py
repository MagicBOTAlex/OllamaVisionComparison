import base64
import os
import re
import httpx
import asyncio
import pandas as pd

resultCsvPath = "./data.csv"
conpairingModel = "llama3.2-vision"
predCol = "llama_pred"

# load or initialize the dataframe
if os.path.exists(resultCsvPath):
    df = pd.read_csv(resultCsvPath)
else:
    df = pd.DataFrame(columns=["image", "actual_gender", "llama_pred", "gemma_pred"])

# collect all image paths under ./parsed/
def get_all_images():
    images = []
    for root, _, files in os.walk("./parsed/"):
        for file in files:
            images.append(os.path.join(root, file))
    return images

all_images = get_all_images()

genderPrompt = """
which gender does the person in the image have?
start by describing the details of the image, then facial features.
when ready, then place the final answer of gender in bold text. (Either male or female)
"""

async def generate_image_text(model: str, image_path: str, prompt: str, stream: bool = False) -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "images": [img_b64]
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "")

async def predict_missing():
    # ensure every image has a row
    for path in all_images:
        if path not in df["image"].values:
            df.loc[len(df)] = [path, None, None, None]

    # now process only those without a llama_pred
    for idx, row in df.iterrows():
        if pd.isna(row[predCol]) or row[predCol] == "":
            print(f"Predicting for: {row['image']}")
            try:
                text = await generate_image_text(conpairingModel, row["image"], genderPrompt)
                
                # take last non-empty line
                lines = [l for l in text.strip().splitlines() if l.strip()]
                last_line = lines[-1] if lines else ""
                
                # regex for male or female
                m = re.search(r"\b(male|female)\b", last_line, re.IGNORECASE)
                gender = m.group(1).lower() if m else "unknown"
                
                # save just the regex result
                df.at[idx, predCol] = gender
                df.to_csv(resultCsvPath, index=False)
                print(f"Saved: {gender}\n")
            except Exception as e:
                print(f"Error on {row['image']}: {e}")
        else:
            print(f"Skipped (already has prediction): {row['image']}")

if __name__ == "__main__":
    asyncio.run(predict_missing())

