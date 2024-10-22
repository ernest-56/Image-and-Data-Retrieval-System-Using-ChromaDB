import torch
from transformers import CLIPProcessor, CLIPModel

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def generate_image_embedding(image, model, processor):
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_embedding = model.get_image_features(**inputs).numpy()
    return image_embedding.tolist()

def generate_text_embedding(text, model, processor):
    inputs = processor(text=text, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embedding = model.get_text_features(**inputs).numpy()
    return text_embedding.tolist()
