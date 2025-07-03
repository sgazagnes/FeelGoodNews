import os
import re
import requests
from datetime import datetime
from huggingface_hub import InferenceClient
from cloudflare import Cloudflare
import base64

# client = InferenceClient(
#     api_key=os.getenv('HUGGING_FACE_TOKEN_NEWS'),
# )

if "CLOUDFLARE_API_TOKEN" in os.environ:
    api_token = os.environ["CLOUDFLARE_API_TOKEN"]


if "CLOUDFLARE_ACCOUNT_ID" in os.environ:
    account_id = os.environ["CLOUDFLARE_ACCOUNT_ID"]


# Initialize client
client = Cloudflare(api_token=api_token)
# output is a PIL.Image object

# output is a PIL.Image object

def generate_image_hf(prompt: str, article_title: str) -> str:
    """
    Generate an image from prompt using Hugging Face InferenceClient (Stable Diffusion)
    """
    try:
        # Generate the image (PIL.Image)
        image = client.text_to_image(prompt)

        # Create safe filename
        safe_title = re.sub(r'[^\w\s-]', '', article_title).strip()
        safe_title = re.sub(r'[-\s]+', '-', safe_title)[:50]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('public/images', exist_ok=True)
        filename = f"public/images/news_image_{safe_title}_{timestamp}.png"

        # Save the PIL image
        image.save(filename)

        print(f"✅ Image saved: {filename}")
        return filename

    except Exception as e:
        print(f"Error generating image: {e}")
        return None


if __name__ == "__main__":
    prompt = "A warm, uplifting photorealistic image of people helping each other in a sunny park"
    title = "People Helping Each Other"

    print("🎨 Generating image...")
    # image_path = generate_image_hf(prompt, title)
#     image = client.text_to_image(
#     "Astronaut riding a horse",
#     model="stabilityai/stable-diffusion-3.5-large",
# )       
    data = client.ai.with_raw_response.run(
        "@cf/black-forest-labs/flux-1-schnell",
        account_id=account_id,
        prompt=prompt,
    ).json() 

    image_base64 = data["result"]["image"]

    # Decode to bytes
    image_bytes = base64.b64decode(image_base64)
    # print(image_bytes)
    # Generate safe filename
    safe_title = "developer-excited-about-ai"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("public/images", exist_ok=True)
    filename = f"public/images/{safe_title}_{timestamp}.png"

    # Write to file
    with open(filename, "wb") as f:
        f.write(image_bytes)

    print(f"✅ Image saved to: {filename}")