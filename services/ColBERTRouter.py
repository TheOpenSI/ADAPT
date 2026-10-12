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
                 required_strength_percentage=None,
                 a=-36.9964,
                 b=111.7766,

                 gate_mode="off",                # "off", "strength" or "mad"

             
                 length_normalise=True,

                 mad_tau=1.0,
                 c0=1.0,
                 c1=1.0,

    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold

        self.required_strength_percentage = required_strength_percentage
        self.a, self.b = a, b

        self.gate_mode = gate_mode
        self.length_normalise = length_normalise

        self.mad_tau, self.c0, self.c1 = mad_tau, c0, c1

        print(f"[ColBERTRouter] Loading ColBERT model '{model_name}'...")
        load_start = time.perf_counter()
        self.model = MultiVectorEncoder(model_name, device="cuda")
        print(f"[ColBERTRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[ColBERTRouter] Loaded {len(self.services)} services.")

        if self.verbose:
            print("  Services loaded:")
            for name, desc in self.services.items():
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")

        print("[ColBERTRouter] Encoding service descriptions...")
        self.names = list(self.services.keys())
        descriptions = list(self.services.values())
        self.document_embeddings = self.model.encode_document(descriptions)

        print("[ColBERTRouter] Ready to route.")

    def route(self, query):

        if self.verbose:
            preview = query if len(query) <= 80 else query[:77] + "..."
            print(f"  Scoring query ({len(query)} chars) against {len(self.services)} documents: {preview!r}")

        # Make sure previous GPU work is finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()

        query_embeddings = self.model.encode_query(query)
        n_query_tokens = query_embeddings.shape[0]

        # MaxSim late-interaction scoring (higher is more relevant)
        # result looks like tensor([[9.05, 10.18, 9.12, 9.39]]), that's why scores[0]
        scores = self.model.similarity(query_embeddings, self.document_embeddings)[0]

        top_label = self.names[int(scores.argmax())]

        scores = [float(s) for s in scores]
        if self.length_normalise:
            scores = [s / n_query_tokens for s in scores]
        scores_dict = dict(zip(self.names, scores))

        confidence = ConfidenceMetrics(scores_dict)
        confidence_result = confidence.calculate()

        top_zscore = float(confidence_result["top_zscore"])
        second_top_zscore = float(confidence_result["second_top_zscore"])
        zscore_margin = float(confidence_result["zscore_margin"])

        service_count = len(scores_dict)

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

        else:
            raise ValueError(f"unknown gate_mode: {self.gate_mode}")

        predicted_label = top_label if admitted else self.fallback_label

        # Wait until all GPU operations are finished
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latency_ms = (time.perf_counter() - start) * 1000

        prompt_tokens = n_query_tokens

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
        }


# if __name__ == "__main__":
#     router = ColBERTRouter(
#         service_desc_path="../data/dataset/service_descriptions_3.json",
#         model_name="mixedbread-ai/mxbai-colbert-large-v1",
#         verbose=True,
#     )
#
#     for query in ["Why is the sky blue?", "Which of the following statements is true?"]:
#         result = router.route(query)
#         print(f"predicted_label = {result['predicted_label']} | top_candidate = {result['top_candidate']}")