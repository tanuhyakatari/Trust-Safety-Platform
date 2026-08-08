import torch
import time
import joblib
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

print("Loading CLIP model for Authenticity Agent...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

reference_embeddings = joblib.load("models/reference_embeddings.pkl")
reference_paths = joblib.load("models/reference_paths.pkl")

def _get_embedding(img: Image.Image):
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

def check_listing(image_path: str, listed_price: float, msrp: float):
    """
    Compares a new listing's image against the reference bank of known
    genuine product photos, and checks price against MSRP.
    Returns a counterfeit risk assessment.
    """
    start = time.time()

    img = Image.open(image_path).convert("RGB")
    listing_emb = _get_embedding(img)

    similarities = torch.cosine_similarity(listing_emb, reference_embeddings)
    max_similarity = similarities.max().item()
    best_match_idx = similarities.argmax().item()
    best_match_path = reference_paths[best_match_idx]

    price_ratio = listed_price / msrp if msrp > 0 else 1.0
    price_flag = price_ratio < 0.4  # more than 60% below MSRP is suspicious
    image_flag = max_similarity < 0.75  # low similarity to known genuine images

    latency_ms = (time.time() - start) * 1000

    if price_ratio < 0.15:
        decision = "block"
        reason = f"Price is only {price_ratio:.0%} of MSRP — decisive on its own regardless of image match, extremely unlikely to be genuine"
    elif image_flag and price_flag:
        decision = "block"
        reason = f"Low image similarity ({max_similarity:.2f}) AND price {price_ratio:.0%} of MSRP — likely counterfeit"
    elif image_flag or price_flag:
        decision = "human_review"
        reason = f"Image similarity {max_similarity:.2f}, price {price_ratio:.0%} of MSRP — needs manual check"
    else:
        decision = "approve"
        reason = f"Image similarity {max_similarity:.2f}, price within normal range — closely matches known genuine listings"

    return {
        "image_similarity": round(max_similarity, 4),
        "closest_reference_match": best_match_path,
        "price_ratio": round(price_ratio, 4),
        "decision": decision,
        "reason": reason,
        "latency_ms": round(latency_ms, 2)
    }

if __name__ == "__main__":
    # Quick test using one of the dataset's own images
    test_image = reference_paths[0]
    result = check_listing(test_image, listed_price=250, msrp=1500)
    print(result)

    # Test with a suspiciously low price on the same image
    result2 = check_listing(test_image, listed_price=100, msrp=1500)
    print(result2)
