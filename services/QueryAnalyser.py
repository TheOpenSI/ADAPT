import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import time
import ollama
import requests
from services.RouterBase import RouterBase


class QueryAnalyser(RouterBase):
    routing_method = "llm"

    def __init__(self, service_desc_path, model_name, verbose=False):
        super().__init__(service_desc_path, model_name)
        self.verbose = verbose

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
            "You will be given a list of available services with short descriptions, "
            "followed by a user query. Decide which single service is best suited to "
            "handle that query, using only the descriptions provided below. "
            "Never assume a service exists beyond what is listed.\n\n"
            "Rules:\n"
            "- Match the query against the core intent of each service description, "
            "not just surface keywords.\n"
            "- If more than one service seems plausible, choose the one whose "
            "description is the more specific and direct match.\n"
            "- Respond with only the service name in the exact form - 'service $name', "
            "where 'name' is the exact name given for that service.\n"
            "- Do not include any explanation or additional text.\n\n"
            f"Services:\n{active_services_joined}\n\n"
            f"Query: \"{query}\"\n\n"
            "Which service should handle this query?"
        )
        if self.verbose:
            print(f"  [PROMPT SENT TO MODEL]\n{prompt}\n")
        return prompt

    def _get_ollama_gpu_memory_gb(self):
        try:
            response = requests.get('http://localhost:11434/api/ps', timeout=5)
            data = response.json()
            for model in data.get("models", []):
                if self.model_name in (model.get("name"), model.get("model")):
                    return model.get("size_vram", 0) / (1024**3)
        except Exception as e:
            print(f"[WARNING] Could not fetch Ollama memory info: {e}")
        return None

    def _match_service(self, raw_label):
        cleaned = raw_label.strip().lower()
        if cleaned.startswith("service "):
            cleaned = cleaned[len("service "):]
        for service_name in self.services.keys():
            if service_name.lower() in cleaned:
                return service_name
        print(f"[QueryAnalyser] UNMATCHED — raw model output: {raw_label!r}")
        return "UNMATCHED"

    def route(self, query):
        if self.verbose:
            preview = query if len(query) <= 80 else query[:77] + "..."
            print(f"  Sending query ({len(query)} chars) to '{self.model_name}': {preview!r}")

        start = time.perf_counter()
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "user", "content": self._build_prompt(query)},
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        raw_label = response["message"]["content"]
        predicted_label = self._match_service(raw_label)

        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        gpu_memory_gb = self._get_ollama_gpu_memory_gb()

        if self.verbose:
            print(f"  raw model output: {raw_label!r}")
            print(f"  -> {predicted_label} | {latency_ms:.1f}ms | "
                  f"{prompt_tokens} prompt tok + {completion_tokens} completion tok = {total_tokens} total tok | "
                  f"{gpu_memory_gb:.2f}GB" if gpu_memory_gb is not None else f"  -> {predicted_label} | {latency_ms:.1f}ms | gpu_memory unavailable")

        return {
            "routing_method": self.routing_method,
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "gpu_memory_gb": gpu_memory_gb, 
        }


# if __name__ == "__main__":
#     analyser = QueryAnalyser(
#         service_desc_path="../data/dataset/service_descriptions_3.json",
#         model_name="qwen2.5:7b",
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