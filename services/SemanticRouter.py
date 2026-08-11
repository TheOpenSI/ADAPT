import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from services.RouterBase import RouterBase


class SemanticRouter(RouterBase):
    def __init__(self,service_desc_path,model_name):
        super().__init__(service_desc_path,model_name)
        print(f"[SemanticRouter] Loading embedding model '{model_name}'... (this may take a while on first run)")
        self.model = SentenceTransformer(model_name) 
        print(f"[SemanticRouter] Model is running on device: {self.model.device}")
        print(f"[SemanticRouter] Model loaded.")
        print(f"[SemanticRouter] Loaded {len(self.services)} services.")

        print(f"[SemanticRouter] Building service embeddings...")
        self.service_embeddings = self._build_service_embeddings()
        print(f"[SemanticRouter] Ready to route.")

    def _build_service_embeddings(self):
        names = list(self.services.keys())
        descriptions = list(self.services.values())
        embedded_vec_desc =  self.model.encode(descriptions)
        return dict(zip(names,embedded_vec_desc))

    @staticmethod 
    def _cosine_similarity(a,b):
        numerator = np.dot(a, b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(numerator / denominator)


    def route(self,query):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()
        q_vec = self.model.encode(query)

        scores = {}
        for name, vec in self.service_embeddings.items():
            score = self._cosine_similarity(q_vec, vec)
            scores[name] = score

        latency_ms = (time.perf_counter() - start) * 1000
        predicted_label = max(scores, key=scores.get)

        gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None

        return {
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "gpu_memory_gb":gpu_memory_gb
        }

# if __name__ == "__main__":
#     router = SemanticRouter(
#         service_desc_path="../data/services_desc.json",  
#         model_name="all-MiniLM-L6-v2",
#     )
#     result = router.route("Find the index of <p> in S_5.")
#     print(result)