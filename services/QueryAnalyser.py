import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import time
import ollama
from services.RouterBase import RouterBase


class QueryAnalyser(RouterBase):
    routing_method = "llm"

    def __init__(self, service_desc_path, model_name, verbose=False):
        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

        self.client = ollama.Client(host="http://localhost:11435")
        self.load_model()

        print(f"[QueryAnalyser] Loading service descriptions from {service_desc_path}...")
        load_start = time.perf_counter()
        print(f"[QueryAnalyser] Loaded {len(self.services)} services in {time.perf_counter() - load_start:.2f}s.")

        if self.verbose:
            print(f"  Services loaded:")
            for name, desc in self.services.items():
                preview = desc if len(desc) <= 80 else desc[:77] + "..."
                print(f"    [{name}] ({len(desc)} chars): {preview!r}")

        print(f"[QueryAnalyser] Ready to route using model: {model_name}")

    def _build_prompt(self, query):
        active_services = [f"- {name}: {desc}" for name, desc in self.services.items()]
        active_services_joined = "\n".join(active_services)

        prompt = (
        "You are a query router for a multi-service assistant system. "
        "You will be given a list of available specialist services with short descriptions, "
        "followed by a user query. Decide which single specialist service is best suited to "
        "handle that query, using only the descriptions provided below.\n\n"

        "Rules:\n"
        "- Match the query against the core intent of each specialist service description, "
        "not just surface keywords.\n"
        "- If more than one specialist service seems plausible, choose the one whose "
        "description is the more specific and direct match.\n"
        "- If the query does not match any of the listed specialist services, "
        "return -1.\n"
        "- For a fallback query, respond exactly 'service -1'.\n"
        "- Never assume a specialist service exists beyond those listed below.\n"
        "- Respond only in the exact form 'service $name'.\n"
        "- For a specialist query, $name must exactly match one of the listed specialist names.\n"
        "- Do not include any explanation or additional text.\n\n"

        f"Specialist services:\n{active_services_joined}\n\n"

        f'Query: "{query}"\n\n'

        "Which service should handle this query?"
        )
        
        if self.verbose:
            print(f"  [PROMPT SENT TO MODEL]\n{prompt}\n")
        return prompt

    def _match_service(self, raw_label):

        cleaned = raw_label.strip().lower()

        if cleaned.startswith("service "):
            cleaned = cleaned[len("service "):].strip()

        # General is an allowed fallback
        if cleaned in ["general", "-1"]:
            return "general"

        # Specialist must exactly match a listed service
        for service_name in self.services.keys():
            if cleaned == service_name.lower():
                return service_name

        print(
            f"[QueryAnalyser] UNMATCHED — "
            f"raw model output: {raw_label!r}"
        )

        return "UNMATCHED"

    def load_model(self):
        print(
            f"[QueryAnalyser] Loading Ollama model "
            f"'{self.model_name}'..."
        )

        load_start = time.perf_counter()

        self.client.generate(
            model=self.model_name,
            prompt="",
            keep_alive=-1,
        )

        print(
            f"[QueryAnalyser] Model loaded in "
            f"{time.perf_counter() - load_start:.2f}s"
        )

    def unload_model(self):
        self.client.chat(
            model=self.model_name,
            messages=[],
            keep_alive=0,
    )
        print(f"[QueryAnalyser] Unloaded {self.model_name}")

    def route(self, query):
        if self.verbose:
            preview = query if len(query) <= 80 else query[:77] + "..."
            print(f"  Sending query ({len(query)} chars) to '{self.model_name}': {preview!r}")

        start = time.perf_counter()
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": self._build_prompt(query)},
            ],
        )
        
        raw_label = response["message"]["content"]
        predicted_label = self._match_service(raw_label)

        latency_ms = (time.perf_counter() - start) * 1000


        if self.verbose:
            print(f"  raw model output: {raw_label!r}")
            print(f"  -> {predicted_label} | {latency_ms:.1f}ms")

        return {
            "routing_method": self.routing_method,
            "query": query,
            "raw_response": raw_label,
 
            "predicted_label": predicted_label,
            "top_candidate": predicted_label,
 
            "model_name": self.model_name,
 
            "latency_ms": latency_ms,
 
            "max_score": None,
            "median_score": None,
            **{f"{name}_score": None for name in self.services},
 
            "apply_threshold": None,
 
            "gap": None,
            "gap_cutoff": None,
            "gap_passed": None,
        }


# if __name__ == "__main__":
#     analyser = QueryAnalyser(
#         service_desc_path="../data/dataset/service_descriptions_3.json",
#         model_name="qwen2.5:14b",
#         verbose=True,
#     )

#     test_queries = [
#         "Find the degree for the given field extension Q(sqrt(2), sqrt(3)) over Q.",
#         "Why is the sky blue?",
#         "Which of the following statements is true?",
#     ]

#     for query in test_queries:
#         print(f"\n--- query: {query[:60]}... ---")
#         result = analyser.route(query)
#         print(f"result dict: {result}")