import time
import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import torch
from services.RouterBase import RouterBase
from sentence_transformers import CrossEncoder


class CrossEncoderRouter(RouterBase):
    def __init__(self, service_desc_path, model_name):
        super().__init__(service_desc_path, model_name)
        print(f"[CrossEncoderRouter] Loading cross-encoder model '{model_name}'...")
        self.model = CrossEncoder(model_name)
        print(f"[CrossEncoderRouter] Model loaded.")
        print(f"[CrossEncoderRouter] Model is running on device: {self.model.device}")
        print(f"[CrossEncoderRouter] Loaded {len(self.services)} services.")
        print(f"[CrossEncoderRouter] Ready to route.")


    def route(self,query):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # make a pair of query and service description
        start = time.perf_counter()

        names = list(self.services.keys())
        pairs = [(query,service_desc) for service_desc in self.services.values()]
        scores = self.model.predict(pairs) 

        latency_ms = (time.perf_counter() - start) * 1000

        # get the best one and rename
        best_idx = scores.argmax()
        predicted_label = names[best_idx]

        gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None

        return {
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "gpu_memory_gb": gpu_memory_gb
        }


# if __name__=="__main__":
#     analyser = CrossEncoderRouter(service_desc_path="../data/services_desc.json", model_name="cross-encoder/ms-marco-MiniLM-L6-v2")
