import os
import sys
 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
from sentence_transformers import CrossEncoder
 
from services.confidence_checker.metrics import ConfidenceMetrics
from services.RouterBase import RouterBase
from services.auto_calibration.zref_gate import ZReferenceGate
 
class CrossEncoderRouter(RouterBase):
    routing_method = "cross-encoder"
 
    def __init__(self, 
                 service_desc_path, 
                 model_name, 
                 verbose=False,
                 
                 fallback_label="general",
                 apply_threshold=True,
                 required_strength_percentage=None,
                 a=-39.6013, 
                 b=93.3028,
                 
                 reference_queries=None,
                 percentile=10,
                 gate_mode="shield",               # "shield", "strength", "both" or "off"
    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold

        self.required_strength_percentage = required_strength_percentage
        self.a, self.b = a, b

        self.gate_mode = gate_mode

 
        print(f"[CrossEncoderRouter] Loading cross-encoder model '{model_name}'...")
        load_start = time.perf_counter()
        self.model = CrossEncoder(model_name, device="cuda")
        print(f"[CrossEncoderRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[CrossEncoderRouter] Loaded {len(self.services)} services.")

        # gate goes here, after the model is loaded
        if reference_queries is None:
            raise ValueError("CrossEncoderRouter needs reference_queries to calibrate the gate")

        print(f"[CrossEncoderRouter] Calibrating shield: {len(reference_queries)} reference queries x {len(self.services)} specialists...")
        gate_start = time.perf_counter()

        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_calibration", "cache")

        self.gate = ZReferenceGate(
            model_name=model_name,
            services=self.services,
            reference_queries=reference_queries,
            scorer=self.score_queries,
            percentile=percentile,
            cache_dir=cache_dir,
        )

        print(f"[CrossEncoderRouter] Shield ready in {time.perf_counter() - gate_start:.2f}s")
        print(f"  percentile: {percentile} (about {percentile}% of off-topic queries pass)")
        print(f"  cutoff:     {self.gate.cutoff:.4f}")
        print(f"  cache file: {self.gate.cache_file}")

 
        if self.verbose:
            print(f"  Services loaded:")
            for name, desc in self.services.items():
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")
 
        print(f"[CrossEncoderRouter] Ready to route.")


    def score_queries(self, model_name, queries, description):
        """Score many queries against ONE description. Used by the gate to calibrate."""
        pairs = [(query, description) for query in queries]
        return [float(s) for s in self.model.predict(pairs)]
 
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

        decision = self.gate.decide(scores_dict, fallback=self.fallback_label)
        
        shield_ok = decision["passed"] 
        strength_ok = strength_result["strength_passed"] 


        if not self.apply_threshold or self.gate_mode == "off":
            admitted = True # no gate: always the top specialist
        
        elif self.gate_mode == "shield":
            admitted = shield_ok       

        elif self.gate_mode == "strength":
            admitted = strength_ok           

        elif self.gate_mode == "both":
            admitted = shield_ok and strength_ok

        else:
            raise ValueError(f"unknown gate_mode: {self.gate_mode}")

        predicted_label = top_label if admitted else self.fallback_label
    

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
                        "z_max": strength_result["z_max"],
                        
                        "winner_strength_percentage": (strength_result["winner_strength_percentage"]),
        
                        "required_strength_percentage": (strength_result["required_strength_percentage"]),
                        
                        "automatic_threshold": (strength_result["automatic_threshold"]),
                        "strength_passed": (strength_result["strength_passed"]),
        
                        "apply_threshold": self.apply_threshold,
                        
                        "gate_mode": self.gate_mode,
                        "lead": decision["lead"],
                        "lead_cutoff": decision["cutoff"],
                        "shield_passed": shield_ok,


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