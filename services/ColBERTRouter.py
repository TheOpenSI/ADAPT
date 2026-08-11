import time
import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import torch
from services.RouterBase import RouterBase
from pylate import models, rank

class ColBERTRouter(RouterBase):
    def __init__(self, service_desc_path, model_name):
        super().__init__(service_desc_path, model_name)

        self.model = models.ColBERT(model_name_or_path="colbert-ir/colbertv2.0",
                                           device="cuda" if torch.cuda.is_available() else "cpu",)

        print(f"[ColBERTRouter] Loading ColBERT model '{model_name}'...")
        print(f"[ColBERTRouter] Model loaded.")

        print(
            f"[ColBERTRouter] Loaded {len(self.services)} services."
        )

        self.service_names = list(
            self.services.keys()
        )

        self.service_descriptions = list(
            self.services.values()
        )

        print(
            "[ColBERTRouter] Encoding service descriptions..."
        )


        self.service_embeddings = self.model.encode(
            self.service_descriptions,
            is_query=False,
            show_progress_bar=False,
        )

        print(
            "[ColBERTRouter] Ready to route."
        )


    def route(self, query):

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start = time.perf_counter()

        # Encode query as ColBERT query vectors
        query_embeddings = self.model.encode(
            [query],
            is_query=True,
            show_progress_bar=False,
        )

        # PyLate rerank expects one candidate list per query
        document_ids = [
            self.service_names
        ]

        document_embeddings = [
            self.service_embeddings
        ]

        ranked_results = rank.rerank(
            documents_ids=document_ids,
            queries_embeddings=query_embeddings,
            documents_embeddings=document_embeddings,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        best_result = ranked_results[0][0]

        predicted_label = best_result["id"]

        gpu_memory_gb = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
            if torch.cuda.is_available()
            else None
        )

        return {
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "gpu_memory_gb": gpu_memory_gb,
        }

if __name__ == "__main__":
    router = ColBERTRouter(
        service_desc_path="../data/services_desc.json",
        model_name="colbert-ir/colbertv2.0",
    )
    result = router.route("Find the index of <p> in S_5.")
    print(result)

