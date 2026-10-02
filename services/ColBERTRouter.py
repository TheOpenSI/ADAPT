import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
from sentence_transformers import MultiVectorEncoder

from services.RouterBase import RouterBase
from services.confidence_checker.metrics import ConfidenceMetrics

class ColBERTRouter(RouterBase):
    routing_method = "colbert"

    def __init__(self, 
                 service_desc_path, 
                 model_name, 
                 verbose=False,
                 fallback_label="general",
                 apply_threshold=True,
    ):
        
        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold
    
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

        if self.verbose:
            preview = query if len(query) <= 80 else query[:77] + "..."
            print(f"  Scoring query ({len(query)} chars) against {len(self.services)} documents: {preview!r}")

        # Make sure previous GPU work is finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()

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

        best_idx = scores.argmax()
        top_label = self.names[best_idx]

        # send scores and names 
        scores = [float(s) for s in scores] 
        scores_dict = dict(zip(self.names, scores))

        confidence = ConfidenceMetrics(scores_dict)
        confidence_result = confidence.calculate()

        top_zscore = float(confidence_result["top_zscore"])
        second_top_zscore = float(confidence_result["second_top_zscore"])
        zscore_margin = float(confidence_result["zscore_margin"])

        service_count = len(scores_dict)

        strength_result = self._calculate_winner_strength(top_zscore=top_zscore,
                                                          service_count=service_count
                                                          )

        if self.apply_threshold:
            predicted_label = (
                top_label
                if strength_result["strength_passed"]
                else self.fallback_label
            )
        else:
            predicted_label = top_label
        

        # Wait until all GPU operations are finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()


        latency_ms = (time.perf_counter() - start) * 1000

        
        prompt_tokens = query_embeddings.shape[0] 

        if self.verbose:
            ranked = sorted(zip(self.names, scores), key=lambda x: x[1], reverse=True)
            print("  scores (ranked):")
            for name, score in ranked:
                marker = " <-- predicted" if name == predicted_label else ""
                print(f"    {name}: {score:.4f}{marker}")
            print(f"  -> {predicted_label} | {latency_ms:.1f}ms | {prompt_tokens} tok")

            print(
                             f" max={confidence_result['max']}"
                             f" | margin={confidence_result['margin']}"
                             f" | top_zscore={top_zscore}"
                             f" | zscore_margin={zscore_margin}"
                        )
                        
            print("\n => Automatic threshold calculation")
            print(f" K: {service_count}")
            print(f" z_max: {strength_result['z_max']:.4f}")
            print(
                 f" required percentage: "
                 f"{strength_result['required_strength_percentage']:.2f}%"
            )
            print(
                 f" calculation: "
                 f"({strength_result['required_strength_percentage']:.2f}/100)"
                 f" * {strength_result['z_max']:.4f}"
            )
            
            print(
                 f" T*(K): "
                 f"{strength_result['automatic_threshold']:.4f}"
            )

            print(f" observed top z-score: {top_zscore:.4f}")
            print(
                f" passed threshold: "
                f"{strength_result['strength_passed']}"
            )

        
        return {
                                "routing_method": self.routing_method,
                                "query": query,
                    
                                "raw_response": None,
                                "predicted_label": predicted_label,
                                "top_candidate": top_label,
                    
                                "model_name": self.model_name,
                    
                                "latency_ms": latency_ms,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": 0,
                                "total_tokens": prompt_tokens,
                
                
                                # Raw score metrics
                                "max_score": confidence_result["max"],
                                "margin": confidence_result["margin"],
                    
                                "mean_score": confidence_result["mean"],
                                "std_score": confidence_result["std"],
                
                                # Z-score metrics
                                "top_zscore": confidence_result["top_zscore"],
                                "second_top_zscore": confidence_result["second_top_zscore"],
                                "zscore_margin": confidence_result["zscore_margin"],
                
                                # Per-service details
                                "scores": scores_dict,
                                "zscores": confidence_result["zscores"],
                
                                # Automatically fitted winner-strength gate
                                "service_count": service_count,
                                "z_max": strength_result["z_max"],
                                
                                "winner_strength_ratio": (strength_result["winner_strength_ratio"]),
                                "winner_strength_percentage": (strength_result["winner_strength_percentage"]),
                
                                "required_strength_percentage": (strength_result["required_strength_percentage"]),
                                
                                "automatic_threshold": (strength_result["automatic_threshold"]),
                                "strength_passed": (strength_result["strength_passed"]),
                
                                "apply_threshold": self.apply_threshold,
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