# utils/zero_shot.py
import torch
import torch.nn.functional as F

# You can edit/extend this list anytime.
DEFAULT_CATEGORIES = [
    "game", "shopping", "finance", "food", "travel",
    "fitness", "education", "utility", "music", "photo",
    "kids", "entertainment", "productivity", "social",
    "beauty", "automotive", "health"
]

def clip_zero_shot_labels(
    clip_model, clip_processor, pil_image,
    categories=DEFAULT_CATEGORIES, top_k=3, device=None
):
    """
    Returns top-k zero-shot labels with cosine similarity scores using CLIP.
    - clip_model: SentenceTransformer CLIP (your existing model)
    - clip_processor: the same SentenceTransformer (provides .encode on text)
    - pil_image: PIL image (RGB)
    """
    # SentenceTransformers API gives .encode for both text and images
    # Make sure we stay on CPU/MPS the same way you use elsewhere
    img_feat = clip_processor.encode(pil_image, convert_to_tensor=True, normalize_embeddings=True, device=device)
    text_prompts = [f"a mobile ad about {c}" for c in categories]
    text_feat = clip_processor.encode(text_prompts, convert_to_tensor=True, normalize_embeddings=True, device=device)

    # Cosine similarity
    sims = (img_feat @ text_feat.T).squeeze(0).tolist()
    ranked = sorted(zip(categories, sims), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"label": c, "score": round(float(s), 3)} for c, s in ranked]
