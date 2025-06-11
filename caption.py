"""
Image Captioning with BLIP - Hugging Face
Use a local image or URL to generate a caption in English.
"""

import argparse
from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_source, is_url=False):
    """Generate caption from image path or URL."""
    raw_image = Image.open(requests.get(image_source, stream=True).raw, mode='r').convert("RGB") if is_url else Image.open(image_source).convert("RGB")
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    print(f"Caption: {caption}")
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Captioning using BLIP.")
    parser.add_argument("--image", type=str, help="Path to local image file.")
    parser.add_argument("--url", type=str, help="URL of the image.")
    args = parser.parse_args()

    if args.image:
        generate_caption(args.image)
    elif args.url:
        generate_caption(args.url, is_url=True)
    else:
        print("Please provide --image or --url.")
