import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")
import torch
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from services.RouterBase import RouterBase
from services.confidence_checker.metrics import ConfidenceMetrics


class SemanticRouter(RouterBase):
    routing_method = "bi-encoder"

    def __init__(self, 
                 service_desc_path,
                 model_name,
                 verbose=False,

                 required_strength_percentage = 80,
                 fallback_label="general",
                 apply_threshold=True,

    ):
        
        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.required_strength_percentage = (float(required_strength_percentage))
        self.required_strength_ratio = (self.required_strength_percentage / 100)

        self.fallback_label = fallback_label
        self.apply_threshold = apply_threshold

        print(f"[SemanticRouter] Loading embedding model '{model_name}'... (this may take a while on first run)")
        load_start = time.perf_counter()
        self.model = SentenceTransformer(model_name).cuda()
        print(f"[SemanticRouter] Model loaded in {time.perf_counter() - load_start:.2f}s — running on device: {self.model.device}")
        print(f"[SemanticRouter] Loaded {len(self.services)} services.")

        print(f"[SemanticRouter] Building service embeddings...")
        self.service_embeddings = self._build_service_embeddings()
        print(f"[SemanticRouter] Ready to route.")


    @staticmethod
    def _calculate_winner_strength(top_zscore,
                                   service_count,
                                   required_ratio):
        
        if not 0.0 <= required_ratio <= 1.0:
            raise ValueError("required_ratio must be between 0 and 1.")

        maximum_top_zscore = np.sqrt(service_count - 1)

        winner_strength_ratio = (float(top_zscore)/ maximum_top_zscore)

        automatic_top_z_threshold = (required_ratio* maximum_top_zscore)

        strength_passed = (float(top_zscore) >= automatic_top_z_threshold)

        return {
            "service_count": int(service_count),
            "maximum_top_zscore": float(maximum_top_zscore),
            "winner_strength_ratio": float(winner_strength_ratio),
            "winner_strength_percentage": float(winner_strength_ratio * 100.0),
            "required_strength_ratio": float(required_ratio),
            "required_strength_percentage": float(required_ratio * 100.0),
            "automatic_top_z_threshold": float(automatic_top_z_threshold),
            "strength_passed": bool(strength_passed),
        }



    def _build_service_embeddings(self):
        names = list(self.services.keys())
        descriptions = list(self.services.values())

        if self.verbose:
            print(f"  Embedding {len(names)} service descriptions:")
            for name, desc in zip(names, descriptions):
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")

        embedded_vec_desc = self.model.encode(descriptions) ## here is where it embedded those descriptions individually in a list

        if self.verbose:
            print(f"  Service embedding matrix shape: {embedded_vec_desc.shape} "
                  f"({embedded_vec_desc.shape[0]} services x {embedded_vec_desc.shape[1]} dims)")
            # flag anything suspicious: all-zero vectors, NaNs, etc.
            for name, vec in zip(names, embedded_vec_desc):
                if np.isnan(vec).any():
                    print(f"    WARNING: '{name}' embedding contains NaN values")
                if np.allclose(vec, 0):
                    print(f"    WARNING: '{name}' embedding is all zeros — description may be empty")

        return dict(zip(names, embedded_vec_desc)) # then it return back dictionary with key: service name and value: embedded vector

    @staticmethod
    def _cosine_similarity(a, b):
        numerator = np.dot(a, b)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        return float(numerator / denominator)

    def route(self, query):
        if self.verbose:
            preview = query if len(query) <= 80 else query[:77] + "..."
            print(f"  Embedding query ({len(query)} chars): {preview!r}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        q_vec = self.model.encode(query)

        # if self.verbose:
        #     print(f"  Query embedding shape: {q_vec.shape} | "
        #           f"norm={np.linalg.norm(q_vec):.4f} | "
        #           f"first 5 dims: {np.round(q_vec[:5], 4).tolist()}")

        scores = {}
        for name, vec in self.service_embeddings.items():
            score = self._cosine_similarity(q_vec, vec)
            scores[name] = score

        top_label = max(scores, key=scores.get)


        # send scores and names in float not containing np.float32
        scores_dict = {name: float(score) for name, score in scores.items()}

        # print("\nDEBUG SCORES")
        # print(scores_dict)

        confidence = ConfidenceMetrics(scores_dict)
        confidence_result = confidence.calculate()

        top_zscore = float(confidence_result["top_zscore"])
        second_top_zscore = float(confidence_result["second_top_zscore"])
        
        zscore_margin = float(confidence_result["zscore_margin"])
        service_count = len(scores_dict)

        strength_result = (self._calculate_winner_strength(top_zscore=top_zscore,
                                                           service_count=service_count,
                                                           required_ratio=(self.required_strength_ratio),
                                                           )
        )



        if self.apply_threshold:
            predicted_label = (
                top_label 
                if strength_result["strength_passed"]
                else self.fallback_label)

        else:
            predicted_label = top_label


        # print("DEBUG ZSCORES")
        # print(confidence_result["zscores"])

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
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            print("  scores (ranked):")
            for name, score in ranked:
                marker = " <-- predicted" if name == predicted_label else ""
                print(f"    {name}: {score:.4f}{marker}")
            print(f"  -> {predicted_label} | {latency_ms:.1f}ms | {prompt_tokens} tok")
            
            print("  winner strength:" f" {strength_result['winner_strength_percentage']:.2f}%")
            print("  required winner strength:" f" {strength_result['required_strength_percentage']:.2f}%")
            print("  maximum possible top z-score:" f" {strength_result['maximum_top_zscore']:.4f}")
            print("  automatic top-z threshold:" f" {strength_result['automatic_top_z_threshold']:.4f}")
            print("  actual top z-score:" f" {top_zscore:.4f}")
            print("  strength passed:"f" {strength_result['strength_passed']}")

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

                # Winner-strength gate
                "service_count": service_count,

                "maximum_top_zscore": (strength_result["maximum_top_zscore"]),
                "winner_strength_ratio": (strength_result["winner_strength_ratio"]),
                "winner_strength_percentage": (strength_result["winner_strength_percentage"]),
                "required_strength_ratio": (strength_result["required_strength_ratio"]),
                "required_strength_percentage": (strength_result["required_strength_percentage"]),
                "automatic_top_z_threshold": (strength_result["automatic_top_z_threshold"]),
                "strength_passed": (strength_result["strength_passed"]),

                "apply_threshold": self.apply_threshold,
            }
    

# if __name__ == "__main__":
#     router = SemanticRouter(
#         service_desc_path="../data/dataset/service_descriptions_5.json",
#         model_name= "BAAI/bge-large-en-v1.5",
#         verbose=True
#         )

#     test_queries = [
#         "Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.",
#         "Why is the sky blue?",
#         "Which of the following statements is true?",  
#     ]

#     for query in test_queries:
#         print(f"\n--- query: {query[:60]}... ---")
#         result = router.route(query)
#         print(f"result dict: {result}")

    

