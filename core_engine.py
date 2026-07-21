import torch
import torchvision.transforms as T
from PIL import Image
import faiss
import numpy as np

class AIVectorEngine:
    def __init__(self):
        # Load DINOv2 ViT-S/14 model for high-resolution pattern & feature extraction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(self.device)
        self.model.eval()

        # Biometric-grade image preprocessing pipeline
        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.index = None
        self.metadata = []

    def get_single_embedding(self, pil_image: Image.Image) -> np.ndarray:
        """Extracts a L2-normalized 384-dimensional visual feature vector."""
        img_t = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img_t)
            # L2 Normalization enables exact Cosine Similarity matching in FAISS / Qdrant
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype("float32")

    def get_batch_embeddings(self, pil_images: list, batch_size: int = 16) -> np.ndarray:
        """Extracts embeddings in batches for high-speed indexing."""
        all_embs = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            tensors = torch.stack([self.transform(img.convert("RGB")) for img in batch]).to(self.device)
            with torch.no_grad():
                embs = self.model(tensors)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                all_embs.append(embs.cpu().numpy().astype("float32"))
        return np.vstack(all_embs)

    def create_index(self, embeddings: np.ndarray, metadata: list):
        """Creates an inner-product (cosine similarity) FAISS index."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata

    def search(self, query_embedding: np.ndarray, top_k: int = 15, min_confidence: float = 0.10) -> list:
        """Searches the vector space and returns ranked visual matches with detailed scoring."""
        if self.index is None or len(self.metadata) == 0:
            return []

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata) and score >= min_confidence:
                results.append({
                    "score": float(score),
                    "meta": self.metadata[idx]
                })
        return results
