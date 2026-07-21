import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "catalog_embeddings"

def get_secret(key_name):
    """Safely retrieves secrets without throwing StreamlitSecretNotFoundError."""
    # 1. Check environment variables (GitHub Actions / Local Terminal / Render)
    val = os.environ.get(key_name)
    if val:
        return val

    # 2. Check Streamlit secrets safely without throwing an exception
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass

    return None


class AIVectorEngine:
    def __init__(self):
        # Neural feature extractor setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Safely fetch Qdrant credentials
        qdrant_url = get_secret("QDRANT_URL")
        qdrant_api_key = get_secret("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("Missing QDRANT_URL or QDRANT_API_KEY. Please verify secrets are configured.")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the Qdrant vector collection if it doesn't already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def get_single_embedding(self, pil_image: Image.Image) -> list:
        img_t = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img_t)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().flatten().tolist()

    def get_batch_embeddings(self, pil_images: list, batch_size: int = 16) -> list:
        all_embs = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            tensors = torch.stack([self.transform(img.convert("RGB")) for img in batch]).to(self.device)
            with torch.no_grad():
                embs = self.model(tensors)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                all_embs.extend(embs.cpu().numpy().tolist())
        return all_embs

    def upsert_points(self, points: list):
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

    def search(self, query_vector: list, top_k: int = 15, min_confidence: float = 0.10) -> list:
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=min_confidence
        )
        
        matches = []
        for res in results:
            matches.append({
                "score": float(res.score),
                "meta": res.payload
            })
        return matches

    def get_all_brands(self) -> list:
        scroll_res, _ = self.client.scroll(collection_name=COLLECTION_NAME, limit=10000, with_payload=True, with_vectors=False)
        companies = set(point.payload.get("company", "General") for point in scroll_res if point.payload)
        return sorted(list(companies))
