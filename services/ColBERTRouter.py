# import os
# import sys
# sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
# import torch
# import time
# from sentence_transformers import MultiVectorEncoder

# from services.RouterBase import RouterBase
# from services.confidence_checker.metrics import ConfidenceMetrics

# from services.entropy_self_calibration.entropy_calibration import EntropyGate

# # The reference queries and the saved scores live in services/entropy_self_calibration/
# HERE = os.path.dirname(os.path.abspath(__file__))
# REFERENCE_CSV = os.path.join(HERE, "entropy_self_calibration", "reference_queries.csv")
# CACHE_DIR = os.path.join(HERE, "entropy_self_calibration", "calibration_cache")


# class ColBERTRouter(RouterBase):
#     routing_method = "colbert"

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
#                  entropy_length_normalise=True,   # ColBERT only: divide scores by the query token count
#                  cache_dir=CACHE_DIR,
#     ):

#         super().__init__(service_desc_path, model_name)
#         self.verbose = verbose

#         self.fallback_label = fallback_label
#         self.apply_threshold = apply_threshold

#         self.required_strength_percentage = required_strength_percentage
#         self.a, self.b = a, b
#         self.use_z_gate = use_z_gate
#         self.entropy_length_normalise = entropy_length_normalise

#         print(f"[ColBERTRouter] Loading ColBERT model '{model_name}'...")
#         load_start = time.perf_counter()
#         self.model = MultiVectorEncoder(model_name, device="cuda")
#         print(f"[ColBERTRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
#         print(f"[ColBERTRouter] Loaded {len(self.services)} services.")

#         if self.verbose:
#             print(f"  Services loaded:")
#             for name, desc in self.services.items():
#                 preview = desc if len(desc) <= 80 else desc[:77] + "..."
#                 print(f"    [{name}] ({len(desc)} chars): {preview!r}")


#         print(f"[ColBERTRouter] Encoding service descriptions...")
#         self.names = list(self.services.keys())
#         descriptions = list(self.services.values())
#         self.document_embeddings = self.model.encode_document(descriptions)

#         # if self.verbose:
#         #     print(f"  Document embeddings — one matrix per service, shape = (num_tokens, embedding_dim):")
#         #     for name, emb in zip(self.names, self.document_embeddings):
#         #         print(f"    [{name}] shape: {emb.shape}  ← {emb.shape[0]} token vectors, {emb.shape[1]} dims each")


#         # Entropy gate: loads cached reference scores, scores only what is
#         # missing (new / edited specialists), then fixes tau and the cutoff.

#         self.entropy_gate = None
#         if use_entropy_gate:
#             if not os.path.exists(reference_queries_path):
#                 raise FileNotFoundError(f"Reference queries not found: {reference_queries_path}")

#             # the saved scores depend on this setting, so it goes in the cache name
#             cache_name = model_name + ("__lennorm" if entropy_length_normalise else "__raw")

#             self.entropy_gate = EntropyGate(
#                 model_name=cache_name,
#                 services=self.services,
#                 reference_queries=self._load_reference_queries(reference_queries_path),

#                 scorer=self._score_one_description,
#                 cache_dir=cache_dir,
#                 percentile=entropy_percentile,
#                 normalise=entropy_normalise,
#                 verbose=self.verbose,
#             )

#         print(f"[ColBERTRouter] Ready to route.")


#     @staticmethod
#     def _load_reference_queries(path):
#         import pandas as pd
#         df = pd.read_csv(path)
#         col = next(c for c in ("question", "query", "text", "utterance") if c in df.columns)
#         return df[col].astype(str).tolist()

#     def _score_one_description(self, model_name, queries, description):
#         """Model-specific part: score many queries against ONE description.
#         Same score as route() (MaxSim). If entropy_length_normalise is on, the
#         score is divided by the query token count, exactly as route() does for
#         the entropy gate."""
#         doc_embedding = self.model.encode_document([description])
#         scores = []
#         for q in queries:
#             q_embedding = self.model.encode_query(q)
#             s = float(self.model.similarity(q_embedding, doc_embedding)[0][0])
#             if self.entropy_length_normalise:
#                 s = s / q_embedding.shape[0]
#             scores.append(s)
#         return scores

#     def route(self, query):

#         if self.verbose:
#             preview = query if len(query) <= 80 else query[:77] + "..."
#             print(f"  Scoring query ({len(query)} chars) against {len(self.services)} documents: {preview!r}")

#         # Make sure previous GPU work is finished
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()

#         start = time.perf_counter()

#         query_embeddings = self.model.encode_query(query)

#         # if self.verbose:
#         #     print(f"  Query embeddings shape: {query_embeddings.shape}  "
#         #           f"← {query_embeddings.shape[0]} token vectors, {query_embeddings.shape[1]} dims each")

#         # MaxSim late-interaction scoring (higher is more relevant)
#         scores = self.model.similarity(query_embeddings, self.document_embeddings)
#         # print("scores",scores)
#         # result: will be this tensor([[9.0512, 10.1842, 9.1238, 9.3910]]), thats why scores[0]
#         scores = scores[0]

#         best_idx = scores.argmax()
#         top_label = self.names[best_idx]

#         # send scores and names
#         scores = [float(s) for s in scores]
#         scores_dict = dict(zip(self.names, scores))

#         confidence = ConfidenceMetrics(scores_dict)
#         confidence_result = confidence.calculate()

#         top_zscore = float(confidence_result["top_zscore"])
#         second_top_zscore = float(confidence_result["second_top_zscore"])
#         zscore_margin = float(confidence_result["zscore_margin"])

#         service_count = len(scores_dict)

#         strength_result = self._calculate_winner_strength(top_zscore=top_zscore,
#                                                                   service_count=service_count,
#                                                                   fixed_percentage=self.required_strength_percentage,
#                                                                   a=self.a,
#                                                                   b=self.b)

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
#                 # MaxSim adds up over query tokens, so a longer query gets bigger
#                 # scores. Divide by the token count for the entropy gate only
#                 # (z and the top pick are not affected by this).
#                 if self.entropy_length_normalise:
#                     gate_scores = {n: s / query_embeddings.shape[0] for n, s in scores_dict.items()}
#                 else:
#                     gate_scores = scores_dict

#                 gate_result = self.entropy_gate.decide(
#                     gate_scores, fallback=self.fallback_label
#                 )
#                 entropy_ok = gate_result["passed"]

#             if self.use_z_gate:
#                 z_ok = strength_result["strength_passed"]

#         if entropy_ok and z_ok:
#             predicted_label = top_label
#         else:
#             predicted_label = self.fallback_label


#         # Wait until all GPU operations are finished
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()


#         latency_ms = (time.perf_counter() - start) * 1000


#         prompt_tokens = query_embeddings.shape[0]

#         if self.verbose:
#             ranked = sorted(zip(self.names, scores), key=lambda x: x[1], reverse=True)
#             print("  scores (ranked):")
#             for name, score in ranked:
#                 marker = " <-- predicted" if name == predicted_label else ""
#                 print(f"    {name}: {score:.4f}{marker}")
#             print(f"  -> {predicted_label} | {latency_ms:.1f}ms | {prompt_tokens} tok")

#             print(
#                              f" max={confidence_result['max']}"
#                              f" | margin={confidence_result['margin']}"
#                              f" | top_zscore={top_zscore}"
#                              f" | zscore_margin={zscore_margin}"
#                         )

#             print("\n => Automatic threshold calculation")
#             print(f" K: {service_count}")
#             print(f" z_max: {strength_result['z_max']:.4f}")
#             print(
#                  f" required percentage: "
#                  f"{strength_result['required_strength_percentage']:.2f}%"
#             )
#             print(
#                  f" calculation: "
#                  f"({strength_result['required_strength_percentage']:.2f}/100)"
#                  f" * {strength_result['z_max']:.4f}"
#             )

#             print(
#                  f" T*(K): "
#                  f"{strength_result['automatic_threshold']:.4f}"
#             )

#             print(f" observed top z-score: {top_zscore:.4f}")

#             if gate_result is not None:
#                 print(
#                     f" entropy: {gate_result['entropy']:.4f}"
#                     f" (cutoff {gate_result['entropy_cutoff']:.4f})"
#                     f" | gate passed: {gate_result['passed']}"
#                 )

#             print(
#                 f" passed threshold: "
#                 f"{strength_result['strength_passed']}"
#             )


#         return {
#                                 "routing_method": self.routing_method,
#                                 "query": query,

#                                 "raw_response": None,
#                                 "predicted_label": predicted_label,
#                                 "top_candidate": top_label,

#                                 "model_name": self.model_name,

#                                 "latency_ms": latency_ms,
#                                 "prompt_tokens": prompt_tokens,
#                                 "completion_tokens": 0,
#                                 "total_tokens": prompt_tokens,


#                                 # Raw score metrics
#                                 "max_score": confidence_result["max"],
#                                 "margin": confidence_result["margin"],

#                                 "mean_score": confidence_result["mean"],
#                                 "std_score": confidence_result["std"],

#                                 # Z-score metrics
#                                 "top_zscore": confidence_result["top_zscore"],
#                                 "second_top_zscore": confidence_result["second_top_zscore"],
#                                 "zscore_margin": confidence_result["zscore_margin"],

#                                 # Per-service details
#                                 "scores": scores_dict,
#                                 "zscores": confidence_result["zscores"],

#                                 # Automatically fitted winner-strength gate
#                                 "service_count": service_count,
#                                 "z_max": strength_result["z_max"],

#                                 "winner_strength_ratio": (strength_result["winner_strength_ratio"]),
#                                 "winner_strength_percentage": (strength_result["winner_strength_percentage"]),

#                                 "required_strength_percentage": (strength_result["required_strength_percentage"]),

#                                 "automatic_threshold": (strength_result["automatic_threshold"]),
#                                 "strength_passed": (strength_result["strength_passed"]),

#                                 "apply_threshold": self.apply_threshold,

#                                 # Entropy gate (None when it is not used)
#                                 "entropy": gate_result["entropy"] if gate_result else None,
#                                 "entropy_cutoff": gate_result["entropy_cutoff"] if gate_result else None,
#                                 "entropy_tau": self.entropy_gate.tau if self.entropy_gate else None,
#                                 "gate_passed": gate_result["passed"] if gate_result else None,
#                             }


# # if __name__ == "__main__":
# #     router = ColBERTRouter(
# #         service_desc_path="../data/dataset/service_descriptions_3.json",
# #         model_name="mixedbread-ai/mxbai-colbert-large-v1",
# #         verbose=True
# #         )

# #     test_queries = [
# #         "Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.",
# #         "Why is the sky blue?",
# #         "Which of the following statements is true?",
# #     ]

# #     for query in test_queries:
# #         print(f"\n--- query: {query[:60]}... ---")
# #         result = router.route(query)
# #         print(f"result dict: {result}")