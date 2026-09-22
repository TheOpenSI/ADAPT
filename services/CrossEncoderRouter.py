import os
import sys
 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
from sentence_transformers import CrossEncoder
 
from services.confidence_checker.metrics import ConfidenceMetrics
from services.RouterBase import RouterBase
 
class CrossEncoderRouter(RouterBase):
    routing_method = "cross-encoder"
 
    def __init__(self, 
                 service_desc_path, 
                 model_name, 
                 verbose=False,
                 
                 confidence_threshold=1.3,
                 fallback_label="general",
                 apply_threshold=True,
    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose


        self.confidence_threshold = confidence_threshold
        self.fallback_label = fallback_label

        self.apply_threshold = apply_threshold

 
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

        best_idx = raw_scores.argmax()
        top_label = names[best_idx]


        # Get raw scores
        scores = [float(s) for s in raw_scores]
        scores_dict = dict(zip(names, scores))

        confidence = ConfidenceMetrics(scores_dict)
        confidence_result = confidence.calculate()

        # print("\nDEBUG SCORES")
        # print(scores_dict)

        zscore_margin = confidence_result["zscore_margin"]


        if self.apply_threshold:
            if zscore_margin >= self.confidence_threshold:
                predicted_label = top_label
            else:
                predicted_label = self.fallback_label
        else:
            predicted_label = top_label


        # print("DEBUG ZSCORES")
        # print(confidence_result["zscores"])

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
                f" max={confidence_result['max']}"
                f" | margin={confidence_result['margin']}"
                f" | top_zscore={confidence_result['top_zscore']}"
                f" | zscore_margin={confidence_result['zscore_margin']}"
                f" | threshold={self.confidence_threshold}"

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

            "max_score": confidence_result["max"],
            "margin": confidence_result["margin"],

            "mean_score": confidence_result["mean"],
            "std_score": confidence_result["std"],

            "top_zscore": confidence_result["top_zscore"],
            "second_top_zscore": confidence_result["second_top_zscore"],
            "zscore_margin": confidence_result["zscore_margin"],

            "scores": scores_dict,
            "zscores": confidence_result["zscores"],

            "confidence_threshold": self.confidence_threshold,
            "apply_threshold": self.apply_threshold,
        }


