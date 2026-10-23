import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
from sentence_transformers import CrossEncoder

from services.RouterBase import RouterBase
import numpy as np


class CrossEncoderRouter(RouterBase):
    routing_method = "cross-encoder"

    def __init__(self,
                 service_desc_path,
                 model_name,
                 verbose=False,

                 fallback_label="general",
                 apply_threshold=True,       # False -> always route to the top specialist

                 gap_c0=None,                # gap threshold T(k) = gap_c0 + gap_c1 * ln(k)
                 gap_c1=None,
    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold

        self.gap_c0, self.gap_c1 = gap_c0, gap_c1

        if apply_threshold and None in (gap_c0, gap_c1):
            raise ValueError(
                "apply_threshold=True needs gap_c0 and gap_c1 fitted for this cross-encoder model."
            )

        print(f"[CrossEncoderRouter] Loading cross-encoder model '{model_name}'...")
        load_start = time.perf_counter()
        self.model = CrossEncoder(model_name, device="cuda")
        print(f"[CrossEncoderRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[CrossEncoderRouter] Loaded {len(self.services)} services.")

        if self.verbose:
            print("  Services loaded:")
            for name, desc in self.services.items():
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")

        print("[CrossEncoderRouter] Ready to route.")

    def score_queries(self, model_name, queries, description):
        """Score many queries against ONE description. Used to calibrate the gate."""
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

        top_label = names[int(raw_scores.argmax())]

        scores = [float(s) for s in raw_scores]
        scores_dict = dict(zip(names, scores))

        # Gap gate: accept the top specialist if max - median > T(k)
        gap = max(scores) - float(np.median(scores))

        if self.apply_threshold:
            gap_cutoff = float(self.gap_c0 + self.gap_c1 * np.log(len(scores)))
            admitted = gap > gap_cutoff
        else:                            # gate off, used when collecting scores
            gap_cutoff = None
            admitted = True

        predicted_label = top_label if admitted else self.fallback_label

        # ---- timed region ends here: scoring + gap decision only ----
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) * 1000

        if self.verbose:
            ranked = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            print("  scores (ranked):")
            for name, score in ranked:
                marker = " <-- top" if name == top_label else ""
                print(f"    {name}: {score:.4f}{marker}")
            print(f"  -> routed to: {predicted_label} | {latency_ms:.1f}ms")
            print(f"  gap={gap:.4f} | cutoff={gap_cutoff} | admitted={admitted}")

        return {
            "routing_method": self.routing_method,
            "query": query,

            "raw_response": None,
            "predicted_label": predicted_label,
            "top_candidate": top_label,

            "model_name": self.model_name,

            "latency_ms": latency_ms,

            "max_score": max(scores),
            "median_score": float(np.median(scores)),
            # one column per specialist, e.g. "anatomy_score"
            **{f"{name}_score": score for name, score in scores_dict.items()},

            "apply_threshold": self.apply_threshold,

            # Gap gate
            "gap": gap,
            "gap_cutoff": gap_cutoff,
            "gap_passed": admitted,
        }