import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
from sentence_transformers import MultiVectorEncoder

from services.RouterBase import RouterBase
import numpy as np


class ColBERTRouter(RouterBase):
    routing_method = "colbert"

    def __init__(self,
                 service_desc_path,
                 model_name,
                 verbose=False,

                 fallback_label="general",
                 apply_threshold=True,       # False -> always route to the top specialist
                 length_normalise=True,

                 gate="gap",

                 gap_c0=None,                # gap threshold T(k) = gap_c0 + gap_c1 * ln(k)
                 gap_c1=None,
                 
                 margin_c0=None,
                 margin_c1=None,
                 max_c0=None,                # max threshold    T(k) = max_c0 + max_c1 * ln(k)
                 max_c1=None,
                 zscore_c0=None,             # z-score threshold T(k) = zscore_c0 + zscore_c1 * ln(k)
                 zscore_c1=None,

                 fixed_threshold=None,       # ablation
    ):

        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold
        self.length_normalise = length_normalise

        self.gap_c0, self.gap_c1 = gap_c0, gap_c1
        self.fixed_threshold = fixed_threshold

        if gate not in ("gap", "margin", "max", "zscore"):
            raise ValueError("gate must be 'gap', 'margin', 'max' or 'zscore'.")
        
        self.gate = gate
        self.margin_c0, self.margin_c1 = margin_c0, margin_c1
        self.max_c0, self.max_c1 = max_c0, max_c1
        self.zscore_c0, self.zscore_c1 = zscore_c0, zscore_c1
        # constants of the active gate
        self.c0, self.c1 = {"gap": (gap_c0, gap_c1), "margin": (margin_c0, margin_c1),
                            "max": (max_c0, max_c1), "zscore": (zscore_c0, zscore_c1)}[gate]

        if apply_threshold and fixed_threshold is None and None in (self.c0, self.c1):
            raise ValueError(
                f"apply_threshold=True with gate='{gate}' needs {gate}_c0 and {gate}_c1 (or fixed_threshold) for this ColBERT model."
                )

        # Which gate is active
        if not apply_threshold:
            gate_mode = "OFF (always route to the top specialist)"
        elif fixed_threshold is not None:
            gate_mode = f"ABLATION, {gate} gate, fixed threshold = {fixed_threshold}"
        else:
            gate_mode = f"{gate} gate, T(k) = {self.c0} + {self.c1} * ln(k)"
        print(f"[ColBERTRouter] Gate: {gate_mode}")

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

        # Gap gate: accept the top specialist if max - median > T(k)
        gap = max(scores) - float(np.median(scores))
        ranked_scores = sorted(scores, reverse=True)
        margin = ranked_scores[0] - ranked_scores[1]
        zscore = (ranked_scores[0] - float(np.mean(scores))) / (float(np.std(scores)) + 1e-9)   # 1e-9 just to handle 0 division
        gate_value = {"gap": gap, "margin": margin, "max": ranked_scores[0], "zscore": zscore}[self.gate]

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

            # z-score value, and the active gate whatever it is (max is in "max_score")
            "zscore": zscore,
            "gate_value": gate_value,
            "gate_cutoff": gate_cutoff,
            "gate_passed": admitted,
        }