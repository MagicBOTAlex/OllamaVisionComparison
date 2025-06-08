import base64
import httpx
import asyncio

# Get gender from image, with changing ethnicity
genderPromt = """
which gender does the person in the image have?
start by describing the details of the image, then facial features.
when ready, then place the final answer of gender in bold text. (Either male or female)
"""

async def generate_image_text(model: str, image_path: str, prompt: str = "What is in this picture?", stream: bool = False) -> str:
    """
    Send an image and return only the model’s textual reply.
    """
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "images": [image_data]
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post("http://localhost:11434/api/generate", json=payload, timeout=60) # Has 60 seconds to respond
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")

# Example usage
if __name__ == "__main__":
    text = asyncio.run(
        generate_image_text(
            "llama3.2-vision",
            "./parsed/asian/n000009.jpg",
            genderPromt
        )
    )
    print(text)
