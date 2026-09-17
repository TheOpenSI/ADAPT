import os
import sys
 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
from sentence_transformers import CrossEncoder
 
from services.confidence_checker.metrics import ConfidenceMetrics
from services.confidence_checker.config import CONFIDENCE_LEVELS
from services.RouterBase import RouterBase
 
class CrossEncoderRouter(RouterBase):
    routing_method = "cross-encoder"
 
    def __init__(self, 
                 service_desc_path, 
                 model_name, 
                 verbose=False, 
                 confidence_c=None, 
                 fallback_label="general"):



        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        if confidence_c is None:
            confidence_c = CONFIDENCE_LEVELS["standard"]
        self.confidence_c = confidence_c

        self.fallback_label = fallback_label
 
        print(f"[CrossEncoderRouter] Loading cross-encoder model '{model_name}'...")
        load_start = time.perf_counter()
        self.model = CrossEncoder(model_name, device="cuda")
        print(f"[CrossEncoderRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[CrossEncoderRouter] Loaded {len(self.services)} services.")
 
        if self.verbose:
            print(f"  Services loaded:")
            for name, desc in self.services.items():
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")
 
        print(f"[CrossEncoderRouter] Ready to route.")
 
    def route(self, query):
 
        if self.verbose:
            preview = query if len(query) <= 80 else query[:77] + "..."
            print(f"  Scoring query ({len(query)} chars) against {len(self.services)} pairs: {preview!r}")


        # Before starting the timer, wait for any previous GPU work to finish
        if torch.cuda.is_available():
            torch.cuda.synchronize()
 
        start = time.perf_counter()
 
        names = list(self.services.keys())
        pairs = [(query, service_desc) for service_desc in self.services.values()]

        raw_scores = self.model.predict(pairs)

        # winner 
        best_idx = raw_scores.argmax()
        top_label = names[best_idx]

        # Get raw scores
        scores = [float(s) for s in raw_scores]
        scores_dict = dict(zip(names, scores))

        confidence = ConfidenceMetrics(scores_dict)
        confidence_result = confidence.calculate(c=self.confidence_c)

        if confidence_result["is_confident"]:
            predicted_label = top_label
        else:
            predicted_label = self.fallback_label

        # wait for GPU to stop properly first before stopping timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
 
        latency_ms = (time.perf_counter() - start) * 1000
 
        prompt_tokens = 0
 
        for q, d in pairs:
            encoded = self.model.tokenizer.encode(
                q,
                d,
                truncation=True,
                max_length=self.model.max_seq_length,
                add_special_tokens=True,
            )
 
            prompt_tokens += len(encoded)
 
        if self.verbose:
            ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
            print("  scores (ranked):")
            for name, score in ranked:
                marker = " <-- top" if name == top_label else ""
                print(f"    {name}: {score:.4f}{marker}")
                
            print(
                f" top_zscore={confidence_result['top_zscore']} "
                f"| threshold={self.confidence_c} "
                f"| confident={confidence_result['is_confident']}"
            )

            print(f"  -> routed to: {predicted_label} | {latency_ms:.1f}ms | {prompt_tokens} tok")
 
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
            "scores": scores_dict,
            "confidence": confidence_result
        }



# if __name__ == "__main__":

#     router = CrossEncoderRouter(
#         service_desc_path="../data/dataset/service_descriptions_3.json",
#         model_name="Qwen/Qwen3-Reranker-4B",
#         verbose=True,
#         confidence_c=CONFIDENCE_LEVELS["strict"]
#     )

#     queries = [
#         "Find the degree for the field extension Q(sqrt(2), sqrt(3)) over Q.",
#         "Why is the sky blue?",
#         "Which of the following statements is true?",
#         "Heart valve replacement surgery",
#     ]

#     for query in queries:
#         print(f"\n--- Query: {query} ---")

#         result = router.route(query)

#         print(f"Predicted: {result['predicted_label']}")
#         print(f"Confidence: {result['confidence']['top_zscore']}")
#         print(f"Confident?: {result['confidence']['is_confident']}")