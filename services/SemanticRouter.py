# import os
# import sys
# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
# import torch
# import time
# import numpy as np
# from sentence_transformers import SentenceTransformer

# from services.RouterBase import RouterBase
# from services.confidence_checker.metrics import ConfidenceMetrics

# from services.auto_calibration.zref_gate import EntropyGate

# # The reference queries and the saved scores live in services/entropy_self_calibration/
# HERE = os.path.dirname(os.path.abspath(__file__))
# REFERENCE_CSV = os.path.join(HERE, "entropy_self_calibration", "reference_queries.csv")
# CACHE_DIR = os.path.join(HERE, "entropy_self_calibration", "calibration_cache")


# class SemanticRouter(RouterBase):
#     routing_method = "bi-encoder"

#     def __init__(self,
#                  service_desc_path,
#                  model_name,
#                  verbose=False,

#                  fallback_label="general",
#                  apply_threshold=True,

#                  required_strength_percentage=None,
#                  a=-36.9964,
#                  b=111.7766,

#                  # --- gates (turn each one on or off) ---
#                  use_entropy_gate=True,
#                  use_z_gate=False,
#                  reference_queries_path=REFERENCE_CSV,
#                  entropy_percentile=10,
#                  entropy_normalise=True,
#                  cache_dir=CACHE_DIR,

#     ):

#         super().__init__(service_desc_path, model_name)
#         self.verbose = verbose


#         self.fallback_label = fallback_label
#         self.apply_threshold = apply_threshold

#         self.required_strength_percentage = required_strength_percentage
#         self.a, self.b = a, b
#         self.use_z_gate = use_z_gate

#         print(f"[SemanticRouter] Loading embedding model '{model_name}'... (this may take a while on first run)")
#         load_start = time.perf_counter()
#         self.model = SentenceTransformer(model_name).cuda()
#         print(f"[SemanticRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
#         print(f"[SemanticRouter] Loaded {len(self.services)} services.")

#         print(f"[SemanticRouter] Building service embeddings...")
#         self.service_embeddings = self._build_service_embeddings()


#         # Entropy gate: loads cached reference scores, scores only what is
#         # missing (new / edited specialists), then fixes tau and the cutoff.

#         self.entropy_gate = None
#         if use_entropy_gate:
#             if not os.path.exists(reference_queries_path):
#                 raise FileNotFoundError(f"Reference queries not found: {reference_queries_path}")

#             self.entropy_gate = EntropyGate(
#                 model_name=model_name,
#                 services=self.services,
#                 reference_queries=self._load_reference_queries(reference_queries_path),

#                 scorer=self._score_one_description,
#                 cache_dir=cache_dir,
#                 percentile=entropy_percentile,
#                 normalise=entropy_normalise,
#                 verbose=self.verbose,
#             )

#         print(f"[SemanticRouter] Ready to route.")


#     def _build_service_embeddings(self):
#         names = list(self.services.keys())
#         descriptions = list(self.services.values())

#         if self.verbose:
#             print(f"  Embedding {len(names)} service descriptions:")
#             for name, desc in zip(names, descriptions):
#                 preview = desc if len(desc) <= 80 else desc[:77] + "..."
#                 print(f"    [{name}] ({len(desc)} chars): {preview!r}")

#         embedded_vec_desc = self.model.encode(descriptions) ## here is where it embedded those descriptions individually in a list

#         if self.verbose:
#             print(f"  Service embedding matrix shape: {embedded_vec_desc.shape} "
#                   f"({embedded_vec_desc.shape[0]} services x {embedded_vec_desc.shape[1]} dims)")
#             # flag anything suspicious: all-zero vectors, NaNs, etc.
#             for name, vec in zip(names, embedded_vec_desc):
#                 if np.isnan(vec).any():
#                     print(f"    WARNING: '{name}' embedding contains NaN values")
#                 if np.allclose(vec, 0):
#                     print(f"    WARNING: '{name}' embedding is all zeros — description may be empty")

#         return dict(zip(names, embedded_vec_desc)) # then it return back dictionary with key: service name and value: embedded vector

#     @staticmethod
#     def _cosine_similarity(a, b):
#         numerator = np.dot(a, b)
#         denominator = np.linalg.norm(a) * np.linalg.norm(b)
#         return float(numerator / denominator)

#     @staticmethod
#     def _load_reference_queries(path):
#         import pandas as pd
#         df = pd.read_csv(path)
#         col = next(c for c in ("question", "query", "text", "utterance") if c in df.columns)
#         return df[col].astype(str).tolist()

#     def _score_one_description(self, model_name, queries, description):
#         """Model-specific part: score many queries against ONE description.
#         Same score as route(): cosine similarity between the two embeddings."""
#         q_vecs = self.model.encode(queries)
#         d_vec = self.model.encode(description)
#         return [self._cosine_similarity(q, d_vec) for q in q_vecs]

#     def route(self, query):
#         if self.verbose:
#             print(f"\n--- Query: {query} ---")

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         start = time.perf_counter()
#         q_vec = self.model.encode(query)

#         # if self.verbose:
#         #     print(f"  Query embedding shape: {q_vec.shape} | "
#         #           f"norm={np.linalg.norm(q_vec):.4f} | "
#         #           f"first 5 dims: {np.round(q_vec[:5], 4).tolist()}")

#         scores = {}
#         for name, vec in self.service_embeddings.items():
#             score = self._cosine_similarity(q_vec, vec)
#             scores[name] = score

#         top_label = max(scores, key=scores.get)


#         # send scores and names in float not containing np.float32
#         scores_dict = {name: float(score) for name, score in scores.items()}

#         # print("\nDEBUG SCORES")
#         # print(scores_dict)

#         confidence = ConfidenceMetrics(scores_dict)
#         confidence_result = confidence.calculate()

#         top_zscore = float(confidence_result["top_zscore"])
#         second_top_zscore = float(confidence_result["second_top_zscore"])

#         zscore_margin = float(confidence_result["zscore_margin"])
#         service_count = len(scores_dict)

#         # Apply the algorithm
#         strength_result = self._calculate_winner_strength(top_zscore=top_zscore,
#                                                            service_count=service_count,
#                                                            fixed_percentage=self.required_strength_percentage,
#                                                            a=self.a,
#                                                            b=self.b)

#         # Gate modes (apply_threshold=True):
#         #   use_entropy_gate | use_z_gate | mode
#         #   -----------------+------------+------------------------------------------
#         #   True             | False      | entropy only (the final method)
#         #   True             | True       | entropy AND z must both pass
#         #   False            | True       | z only (baseline)
#         #   False            | False      | no gate: always route to the top specialist
#         #   apply_threshold=False turns every gate off.

#         gate_result = None
#         entropy_ok = True
#         z_ok = True

#         if self.apply_threshold:
#             if self.entropy_gate is not None:
#                 gate_result = self.entropy_gate.decide(
#                     scores_dict, fallback=self.fallback_label
#                 )
#                 entropy_ok = gate_result["passed"]

#             if self.use_z_gate:
#                 z_ok = strength_result["strength_passed"]

#         if entropy_ok and z_ok:
#             predicted_label = top_label
#         else:
#             predicted_label = self.fallback_label


#         # print("DEBUG ZSCORES")
#         # print(confidence_result["zscores"])

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         latency_ms = (time.perf_counter() - start) * 1000


#         encoded = self.model.tokenizer.encode(
#             query,
#             truncation=True,
#             max_length=self.model.max_seq_length,
#             add_special_tokens=True,
#         )
#         prompt_tokens = len(encoded)

#         if self.verbose:
#             ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#             print("  scores (ranked):")
#             for name, score in ranked:
#                 marker = " <-- predicted" if name == predicted_label else ""
#                 print(f"    {name}: {score:.4f}{marker}")
#             print(f"  -> {predicted_label} | {latency_ms:.1f}ms | {prompt_tokens} tok")

#             print("\n => Automatic threshold calculation")
#             print(f" K: {service_count}")
#             print(f" z_max: {strength_result['z_max']:.4f}")
#             print(" required percentage: "f"{strength_result['required_strength_percentage']:.2f}%")
#             print(" calculation: "
#                   f"({strength_result['required_strength_percentage']:.2f} / 100) "
#                   f"* {strength_result['z_max']:.4f}"
#             )
#             print(
#                 " T*(K): "
#                 f"{strength_result['automatic_threshold']:.4f}"
#             )
#             print(f" observed top z-score: {top_zscore:.4f}")

#             if gate_result is not None:
#                 print(
#                     f" entropy: {gate_result['entropy']:.4f}"
#                     f" (cutoff {gate_result['entropy_cutoff']:.4f})"
#                     f" | gate passed: {gate_result['passed']}"
#                 )

#             print(
#                 " passed threshold: "
#                 f"{strength_result['strength_passed']}"
#             )

#         return {
#                 "routing_method": self.routing_method,
#                 "query": query,

#                 "raw_response": None,
#                 "predicted_label": predicted_label,
#                 "top_candidate": top_label,

#                 "model_name": self.model_name,

#                 "latency_ms": latency_ms,
#                 "prompt_tokens": prompt_tokens,
#                 "completion_tokens": 0,
#                 "total_tokens": prompt_tokens,


#                 # Raw score metrics
#                 "max_score": confidence_result["max"],
#                 "margin": confidence_result["margin"],

#                 "mean_score": confidence_result["mean"],
#                 "std_score": confidence_result["std"],

#                 # Z-score metrics
#                 "top_zscore": confidence_result["top_zscore"],
#                 "second_top_zscore": confidence_result["second_top_zscore"],
#                 "zscore_margin": confidence_result["zscore_margin"],

#                 # Per-service details
#                 "scores": scores_dict,
#                 "zscores": confidence_result["zscores"],

#                 # Automatically fitted winner-strength gate
#                 "service_count": service_count,
#                 "z_max": strength_result["z_max"],

#                 "winner_strength_ratio": (strength_result["winner_strength_ratio"]),
#                 "winner_strength_percentage": (strength_result["winner_strength_percentage"]),

#                 "required_strength_percentage": (strength_result["required_strength_percentage"]),

#                 "automatic_threshold": (strength_result["automatic_threshold"]),
#                 "strength_passed": (strength_result["strength_passed"]),

#                 "apply_threshold": self.apply_threshold,

#                 # Entropy gate (None when it is not used)
#                 "entropy": gate_result["entropy"] if gate_result else None,
#                 "entropy_cutoff": gate_result["entropy_cutoff"] if gate_result else None,
#                 "entropy_tau": self.entropy_gate.tau if self.entropy_gate else None,
#                 "gate_passed": gate_result["passed"] if gate_result else None,
#             }


# # if __name__ == "__main__":
# #     router = SemanticRouter(
# #         service_desc_path="../data/dataset/service_descriptions_5.json",
# #         model_name= "BAAI/bge-large-en-v1.5",
# #         verbose=True
# #         )

# #     test_queries = [
# #         "Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.",
# #         "Why is the sky blue?",
# #         "Which of the following statements is true?",
# #     ]

# #     for query in test_queries:
# #         result = router.route(query)
# #         print(f"result dict: {result}")