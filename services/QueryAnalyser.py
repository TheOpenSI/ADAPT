import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import time
import ollama
import requests
from services.RouterBase import RouterBase


class QueryAnalyser(RouterBase):
    def __init__(self, service_desc_path, model_name):
        super().__init__(service_desc_path, model_name)
        print(f"[QueryAnalyser] Loading service descriptions from {service_desc_path}...")
        print(f"[QueryAnalyser] Ready to route using model: {model_name}")
        print(f"[QueryAnalyser] Loaded {len(self.services)} services: {list(self.services.keys())}")

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
        # print(f"[PROMPT SENT TO MODEL]\n{prompt}\n")
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

        gpu_memory_gb = self._get_ollama_gpu_memory_gb()

        return {
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "gpu_memory_gb": gpu_memory_gb,
        }

# if __name__ == "__main__":
#     analyser = QueryAnalyser(service_desc_path="../data/services_desc.json", model_name="qwen2.5:7b")
#     result = analyser.route("Find the index of <p> in S_5.")
#     print(result)


