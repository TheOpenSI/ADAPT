from pathlib import Path
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticRouter:
    def __init__(self,service_desc_path,model_name):
        self.service_desc_path = service_desc_path
        self.model = SentenceTransformer(model_name) 
        self.model_name = model_name
        self.services = self._get_service_descriptions() 
        self.service_embeddings = self._build_service_embeddings()

    def _get_service_descriptions(self):
        # change service_desc_path from string to path
        path = Path(self.service_desc_path)
        if not path.exists():
            raise FileNotFoundError(f"Service description file '{self.service_desc_path}' not found.")
        with open(path, "r") as f:
            data = json.load(f)
            return data

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
        start = time.perf_counter()
        q_vec = self.model.encode(query)

        scores = {}
        for name, vec in self.service_embeddings.items():
            score = self._cosine_similarity(q_vec, vec)
            scores[name] = score
            print(scores) 

        latency_ms = (time.perf_counter() - start) * 1000
        predicted_label = max(scores, key=scores.get)

        return {
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

# if __name__ == "__main__":
#     router = SemanticRouter(
#         service_desc_path="../data/services_desc.json",  
#         model_name="all-MiniLM-L6-v2",
#     )
#     result = router.route("Find the index of <p> in S_5.")
#     print(result)