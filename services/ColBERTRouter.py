import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import time
import torch
from sentence_transformers import MultiVectorEncoder

from services.RouterBase import RouterBase


class ColBERTRouter(RouterBase):
    routing_method = "colbert"

    def __init__(self, service_desc_path, model_name, verbose=False):
        super().__init__(service_desc_path, model_name)
        self.verbose = verbose
    
        print(f"[ColBERTRouter] Loading ColBERT model '{model_name}'...")
        load_start = time.perf_counter()
        self.model = MultiVectorEncoder(model_name, device="cuda")
        print(f"[ColBERTRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[ColBERTRouter] Loaded {len(self.services)} services.")

        if self.verbose:
            print(f"  Services loaded:")
            for name, desc in self.services.items():
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")


        print(f"[ColBERTRouter] Encoding service descriptions...")
        self.names = list(self.services.keys())
        descriptions = list(self.services.values())
        self.document_embeddings = self.model.encode_document(descriptions)

        # if self.verbose:
        #     print(f"  Document embeddings — one matrix per service, shape = (num_tokens, embedding_dim):")
        #     for name, emb in zip(self.names, self.document_embeddings):
        #         print(f"    [{name}] shape: {emb.shape}  ← {emb.shape[0]} token vectors, {emb.shape[1]} dims each")
 
        print(f"[ColBERTRouter] Ready to route.")

    def route(self, query):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if self.verbose:
            preview = query if len(query) <= 80 else query[:77] + "..."
            print(f"  Scoring query ({len(query)} chars) against {len(self.services)} documents: {preview!r}")

        start = time.perf_counter()

        query_embeddings = self.model.encode_query(query)

        # if self.verbose:
        #     print(f"  Query embeddings shape: {query_embeddings.shape}  "
        #           f"← {query_embeddings.shape[0]} token vectors, {query_embeddings.shape[1]} dims each")

        # MaxSim late-interaction scoring (higher is more relevant)
        scores = self.model.similarity(query_embeddings, self.document_embeddings)
        # print("scores",scores)
        # result: will be this tensor([[9.0512, 10.1842, 9.1238, 9.3910]]), thats why scores[0]
        scores = scores[0] 

        latency_ms = (time.perf_counter() - start) * 1000

        best_idx = scores.argmax()
        predicted_label = self.names[best_idx]

        gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else None

        prompt_tokens = query_embeddings.shape[0] 

        if self.verbose:
            ranked = sorted(zip(self.names, scores), key=lambda x: x[1], reverse=True)
            print("  scores (ranked):")
            for name, score in ranked:
                marker = " <-- predicted" if name == predicted_label else ""
                print(f"    {name}: {score:.4f}{marker}")
            print(f"  -> {predicted_label} | {latency_ms:.1f}ms | {prompt_tokens} tok | {gpu_memory_gb:.2f}GB")

        return {
            "routing_method": self.routing_method,
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": prompt_tokens,
            "gpu_memory_gb": gpu_memory_gb,
        }


if __name__ == "__main__":
    router = ColBERTRouter(
        service_desc_path="../data/dataset/service_descriptions_3.json",
        model_name="mixedbread-ai/mxbai-colbert-large-v1",
        verbose=True
        )

    test_queries = [
        "Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.",
        "Why is the sky blue?",
        "Which of the following statements is true?",
    ]

    for query in test_queries:
        print(f"\n--- query: {query[:60]}... ---")
        result = router.route(query)
        print(f"result dict: {result}")