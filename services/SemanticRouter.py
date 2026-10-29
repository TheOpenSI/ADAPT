import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from services.RouterBase import RouterBase


class SemanticRouter(RouterBase):
    routing_method = "bi-encoder"

    def __init__(self,
                 service_desc_path,
                 model_name,
                 verbose=False,

                 fallback_label="general",
                 apply_threshold=True,       # False -> always route to the top specialist

                 gate="gap", 

                 gap_c0=None,                # gap threshold T(k) = gap_c0 + gap_c1 * ln(k)
                 gap_c1=None,
                 margin_c0=None,
                 margin_c1=None,

                 fixed_threshold=None,       # ablation

    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold

        self.gap_c0, self.gap_c1 = gap_c0, gap_c1
        self.fixed_threshold = fixed_threshold

        if gate not in ("gap", "margin"):
            raise ValueError("gate must be 'gap' or 'margin'.")

        self.gate = gate
        self.margin_c0, self.margin_c1 = margin_c0, margin_c1

        self.c0, self.c1 = (gap_c0, gap_c1) if gate == "gap" else (margin_c0, margin_c1)


        if apply_threshold and fixed_threshold is None and None in (self.c0, self.c1):
            raise ValueError(
                f"apply_threshold=True with gate='{gate}' needs {gate}_c0 and {gate}_c1 (or fixed_threshold) for this embedding model."
            )

        # Which gate is active
        if not apply_threshold:
            gate_mode = "OFF (always route to the top specialist)"
        elif fixed_threshold is not None:
            gate_mode = f"ABLATION, fixed threshold = {fixed_threshold}"
        else:
            gate_mode = f"{gate} gate, T(k) = {self.c0} + {self.c1} * ln(k)"
        print(f"[SemanticRouter] Gate: {gate_mode}")

        print(f"[SemanticRouter] Loading embedding model '{model_name}'... (this may take a while on first run)")
        load_start = time.perf_counter()
        self.model = SentenceTransformer(model_name, device="cuda")
        print(f"[SemanticRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[SemanticRouter] Loaded {len(self.services)} services.")

        print("[SemanticRouter] Building service embeddings...")
        self.service_embeddings = self._build_service_embeddings()

        print("[SemanticRouter] Ready to route.")

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
        
        scores_dict = {name: self._cosine_similarity(q_vec, vec)
                       for name, vec in self.service_embeddings.items()}

        top_label = max(scores_dict, key=scores_dict.get)
        scores = list(scores_dict.values())

        # Gap gate: accept the top specialist if max - median > T(k)
        gap = max(scores) - float(np.median(scores))
        ranked_scores = sorted(scores, reverse=True)
        margin = ranked_scores[0] - ranked_scores[1]
        gate_value = gap if self.gate == "gap" else margin

        if self.apply_threshold:
            if self.fixed_threshold is not None:
                gate_cutoff = float(self.fixed_threshold)
            else:
                gate_cutoff = float(self.c0 + self.c1 * np.log(len(scores)))
            admitted = gate_value > gate_cutoff
        else:
            gate_cutoff = None
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
            print(f"  {self.gate}={gate_value:.4f} | cutoff={gate_cutoff} | admitted={admitted}")


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
            
            "gate": self.gate,

            # Gap gate (filled only when gate="gap")
            "gap": gap,
            "gap_cutoff": gate_cutoff if self.gate == "gap" else None,
            "gap_passed": admitted if self.gate == "gap" else None,

            # Margin gate (filled only when gate="margin")
            "margin": margin,
            "margin_cutoff": gate_cutoff if self.gate == "margin" else None,
            "margin_passed": admitted if self.gate == "margin" else None,
        }