import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from services.confidence_checker.metrics import ConfidenceMetrics
from services.RouterBase import RouterBase


class SemanticRouter(RouterBase):
    routing_method = "bi-encoder"

    def __init__(self,
                 service_desc_path,
                 model_name,
                 verbose=False,

                 fallback_label="general",
                 apply_threshold=True,
                 required_strength_percentage=None,
                 a=-36.9964,
                 b=111.7766,

                 gate_mode="off",                # "off", "strength" or "mad"

    
                 mad_tau=0.02306,
                 c0 = 1.3033,
                 c1 = 0.9852,
                 
                 gap_c0=None,
                 gap_c1=None,
    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold

        self.required_strength_percentage = required_strength_percentage
        self.a, self.b = a, b

        self.gate_mode = gate_mode

        self.mad_tau, self.c0, self.c1 = mad_tau, c0, c1
        self.gap_c0, self.gap_c1 = gap_c0, gap_c1


        if gate_mode == "mad" and apply_threshold and None in (mad_tau, c0, c1):
            raise ValueError(
                "gate_mode='mad' needs mad_tau, c0 and c1 for this embedding model. "
                "Run the off-topic queries with gate_mode='off', then MADGate, and pass the numbers in."
            )
        if gate_mode == "gap" and apply_threshold and None in (gap_c0, gap_c1):
            raise ValueError(
                "gate_mode='gap' needs gap_c0 and gap_c1 fitted for this embedding model."
            )

        print(f"[SemanticRouter] Loading embedding model '{model_name}'... (this may take a while on first run)")
        load_start = time.perf_counter()
        self.model = SentenceTransformer(model_name, device="cuda")
        print(f"[SemanticRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[SemanticRouter] Loaded {len(self.services)} services.")

        print(f"[SemanticRouter] Building service embeddings...")
        self.service_embeddings = self._build_service_embeddings()

        print(f"[SemanticRouter] Ready to route.")


    def _build_service_embeddings(self):
        names = list(self.services.keys())
        descriptions = list(self.services.values())

        if self.verbose:
            print(f"  Embedding {len(names)} service descriptions:")
            for name, desc in zip(names, descriptions):
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")

        embedded_vec_desc = self.model.encode(descriptions)   # one vector per description

        if self.verbose:
            print(f"  Service embedding matrix shape: {embedded_vec_desc.shape} "
                  f"({embedded_vec_desc.shape[0]} services x {embedded_vec_desc.shape[1]} dims)")
            for name, vec in zip(names, embedded_vec_desc):
                if np.isnan(vec).any():
                    print(f"    WARNING: '{name}' embedding contains NaN values")
                if np.allclose(vec, 0):
                    print(f"    WARNING: '{name}' embedding is all zeros — description may be empty")

        return dict(zip(names, embedded_vec_desc))             # {service name: vector}

    @staticmethod
    def _cosine_similarity(a, b):
        numerator = np.dot(a, b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(numerator / denominator)

    def route(self, query):

        if self.verbose:
            print(f"\n--- Query: {query} ---")

        # Before starting the timer, wait for any previous GPU work to finish
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        q_vec = self.model.encode(query)

        scores = {name: self._cosine_similarity(q_vec, vec)
                  for name, vec in self.service_embeddings.items()}

        top_label = max(scores, key=scores.get)

        # plain floats, not np.float32
        scores_dict = {name: float(score) for name, score in scores.items()}

        confidence = ConfidenceMetrics(scores_dict)
        confidence_result = confidence.calculate()

        top_zscore = float(confidence_result["top_zscore"])
        second_top_zscore = float(confidence_result["second_top_zscore"])
        zscore_margin = float(confidence_result["zscore_margin"])

        service_count = len(scores_dict)

        # Gap gate
        score_values = list(scores_dict.values())
        gap = max(score_values) - float(np.median(score_values))
        gap_cutoff = (self.gap_c0 + self.gap_c1 * np.log(service_count)
                      if self.gap_c0 is not None else None)
        gap_ok = bool(gap > gap_cutoff) if gap_cutoff is not None else True

     
        mad_result = self._calculate_mad_strength(scores_dict, self.mad_tau, self.c0, self.c1)
        

        strength_result = self._calculate_winner_strength(top_zscore=top_zscore,
                                                          service_count=service_count,
                                                          fixed_percentage=self.required_strength_percentage,
                                                          a=self.a,
                                                          b=self.b)

        # Gate modes:
        #   apply_threshold=False or gate_mode="off" -> always route to the top specialist
        #   gate_mode="strength"                     -> winner-strength gate
        #   gate_mode="mad"                          -> MAD gate

        strength_ok = strength_result["strength_passed"]
        mad_ok = mad_result["mad_passed"]

        if not self.apply_threshold or self.gate_mode == "off":
            admitted = True

        elif self.gate_mode == "strength":
            admitted = strength_ok

        elif self.gate_mode == "mad":
            admitted = mad_ok

        elif self.gate_mode == "gap":
            admitted = gap_ok

        else:
            raise ValueError(f"unknown gate_mode: {self.gate_mode}")

        predicted_label = top_label if admitted else self.fallback_label

        # wait for GPU to stop properly first before stopping timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        encoded = self.model.tokenizer.encode(
            query,
            truncation=True,
            max_length=self.model.max_seq_length,
            add_special_tokens=True,
        )
        prompt_tokens = len(encoded)

        if self.verbose:
            ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
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
            print(f" required percentage: {strength_result['required_strength_percentage']:.2f}%")
            print(
                f" calculation: "
                f"({strength_result['required_strength_percentage']:.2f}/100)"
                f" * {strength_result['z_max']:.4f}"
            )
            print(f" T*(K): {strength_result['automatic_threshold']:.4f}")
            print(f" observed top z-score: {top_zscore:.4f}")
            print(f" passed threshold: {strength_result['strength_passed']}")

            if self.gate_mode == "mad" and mad_result["mad_trust"] is not None:
                print(
                    f" mad trust: {mad_result['mad_trust']:.4f}"
                    f" (cutoff {mad_result['mad_cutoff']:.4f})"
                    f" | passed: {mad_result['mad_passed']}"
                )

        return {
            "routing_method": self.routing_method,
            "query": query,
            "a": self.a,
            "b": self.b,

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

            # MAD gate
            "mad_trust": mad_result["mad_trust"],
            "mad_cutoff": mad_result["mad_cutoff"],
            "mad_tau": mad_result["tau"],
            "mad_passed": mad_result["mad_passed"],

            # Gap gate
            "gap": gap,
            "gap_cutoff": gap_cutoff,
            "gap_passed": gap_ok,
        }


# if __name__ == "__main__":
#     router = SemanticRouter(
#         service_desc_path="../data/dataset/service_descriptions_5.json",
#         model_name="BAAI/bge-large-en-v1.5",
#         verbose=True,
#     )
#
#     for query in ["Why is the sky blue?", "Which of the following statements is true?"]:
#         result = router.route(query)
#         print(f"predicted_label = {result['predicted_label']} | top_candidate = {result['top_candidate']}")