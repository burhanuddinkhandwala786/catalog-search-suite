import os
import torch
import torchvision.transforms as T
import faiss
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

class AIVectorEngine:
    def __init__(self, model_size='small'):
        model_name = 'dinov2_vits14' if model_size == 'small' else 'dinov2_vitb14'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.index = None
        self.metadata = []

    def _extract_color_histogram(self, image: Image.Image):
        """Extracts a 64-bin normalized RGB color fingerprint to enforce strict color matching."""
        img_resized = image.convert("RGB").resize((100, 100))
        np_img = np.array(img_resized)
        
        # Calculate color histograms for R, G, B channels
        rhist, _ = np.histogram(np_img[:, :, 0], bins=8, range=(0, 256), density=True)
        ghist, _ = np.histogram(np_img[:, :, 1], bins=8, range=(0, 256), density=True)
        bhist, _ = np.histogram(np_img[:, :, 2], bins=8, range=(0, 256), density=True)
        
        color_vec = np.concatenate([rhist, ghist, bhist])
        return color_vec / (np.linalg.norm(color_vec) + 1e-7)

    def get_single_embedding(self, image: Image.Image):
        """Combines DINOv2 structural features (80%) + RGB color features (20%)."""
        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            deep_emb = self.model(img_tensor)
            deep_emb = deep_emb / torch.norm(deep_emb, p=2, dim=-1, keepdim=True)
            deep_vec = deep_emb.cpu().numpy().flatten()

        color_vec = self._extract_color_histogram(image)
        
        # Concatenate structure + color fingerprints
        combined_vec = np.concatenate([deep_vec * 0.85, color_vec * 0.15])
        combined_vec /= np.linalg.norm(combined_vec)
        return combined_vec.astype('float32')

    def get_batch_embeddings(self, pil_images, batch_size=16):
        all_embeddings = []
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            
            tensors = torch.stack([self.transform(img.convert("RGB")) for img in batch]).to(self.device)
            with torch.inference_mode():
                deep_embs = self.model(tensors)
                deep_embs = deep_embs / torch.norm(deep_embs, p=2, dim=-1, keepdim=True)
                deep_vecs = deep_embs.cpu().numpy()

            for idx, img in enumerate(batch):
                c_vec = self._extract_color_histogram(img)
                comb = np.concatenate([deep_vecs[idx] * 0.85, c_vec * 0.15])
                comb /= np.linalg.norm(comb)
                all_embeddings.append(comb)

        return np.vstack(all_embeddings).astype('float32')

    def create_index(self, embeddings_list, metadata_list):
        embeddings = np.array(embeddings_list).astype('float32')
        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(1)
            
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = metadata_list

    def search(self, query_embedding, top_k=5, min_confidence=0.78):
        """
        min_confidence (0.78 / 78%):
        If a score is below 78%, it is treated as NOT FOUND instead of giving a wrong result.
        """
        if self.index is None or len(self.metadata) == 0:
            return []
            
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)
            
        distances, indices = self.index.search(query_embedding, top_k * 3)
        results = []
        
        for idx, score in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.metadata):
                confidence = float(np.clip(score, 0.0, 1.0))
                
                # STRICT GUARDRAIL: Filter out low-confidence false positives
                if confidence >= min_confidence:
                    results.append({
                        "meta": self.metadata[idx],
                        "score": confidence
                    })
                    
            if len(results) == top_k:
                break
                
        return results