import torch
import time
import random
import glob
import joblib
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

random.seed(42)

def get_embedding(model, processor, img):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        raw_output = model.get_image_features(**inputs)

    if isinstance(raw_output, torch.Tensor):
        emb = raw_output
    elif hasattr(raw_output, "image_embeds"):
        emb = raw_output.image_embeds
    elif hasattr(raw_output, "pooler_output"):
        emb = raw_output.pooler_output
    else:
        raise RuntimeError(f"Unexpected output type: {type(raw_output)}")

    return emb / emb.norm(dim=-1, keepdim=True)

print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_images = glob.glob("data/dataset-chanel-luxury-bags/**/*.jpg", recursive=True)
print(f"Found {len(all_images)} total images")

sample_paths = random.sample(all_images, min(200, len(all_images)))
print(f"Sampling {len(sample_paths)} images for reference bank...")

embeddings = []
valid_paths = []

start = time.time()
for i, path in enumerate(sample_paths):
    try:
        img = Image.open(path).convert("RGB")
        emb = get_embedding(model, processor, img)
        embeddings.append(emb)
        valid_paths.append(path)
    except Exception as e:
        print(f"Skipped {path}: {e}")

    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(sample_paths)}")

print(f"Done in {time.time() - start:.1f}s")
print(f"Successfully embedded {len(embeddings)}/{len(sample_paths)} images")

embeddings_tensor = torch.cat(embeddings, dim=0)
joblib.dump(embeddings_tensor, "models/reference_embeddings.pkl")
joblib.dump(valid_paths, "models/reference_paths.pkl")
print(f"Saved {len(valid_paths)} reference embeddings to models/")
