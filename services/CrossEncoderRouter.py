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
                 
                 fallback_label="general",
                 apply_threshold=True,
                 required_strength_percentage=None,
                 a=-36.9964, 
                 b=111.7766
    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold

        self.required_strength_percentage = required_strength_percentage
        self.a, self.b = a, b

 
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

        top_zscore = float(confidence_result["top_zscore"])
        second_top_zscore = float(confidence_result["second_top_zscore"])
        zscore_margin = float(confidence_result["zscore_margin"])
        
        service_count = len(scores_dict)

        strength_result = self._calculate_winner_strength(top_zscore=top_zscore,
                                                          service_count=service_count,
                                                          fixed_percentage=self.required_strength_percentage,
                                                          a=self.a, 
                                                          b=self.b)
        # Threshold modes:
        #   apply_threshold | required_strength_percentage | mode
        #   ----------------+------------------------------+------------------------------------------
        #   False           | anything                     | off:   always route to the top specialist
        #   True            | a number, e.g. 70            | fixed: calibration sweep
        #   True            | None                         | auto:  fitted curve p*(K)


        if self.apply_threshold:
            predicted_label = (
                top_label
                if strength_result["strength_passed"]
                else self.fallback_label
            )
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
            print(f"  -> routed to: {predicted_label} | {latency_ms:.1f}ms | {prompt_tokens} tok")

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


# if __name__ == "__main__":
#     router = CrossEncoderRouter(
#         service_desc_path="../data/dataset/service_descriptions_5.json",
#         model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
#         verbose=True,
#         apply_threshold=True,
#     )

#     test_queries = [
#         "Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.",
#         "Why is the sky blue?",
#         "Which of the following statements is true?",
#     ]

#     for query in test_queries:
#         result = router.route(query)

#         print("\nRESULT")
#         print(f"predicted_label = {result['predicted_label']}")
#         print(f"top_candidate = {result['top_candidate']}")
#         print(f"top_zscore = {result['top_zscore']:.4f}")
#         print(f"z_max = {result['z_max']:.4f}")
#         print(f"automatic_threshold = {result['automatic_threshold']:.4f}")
#         print(f"strength_passed = {result['strength_passed']}")