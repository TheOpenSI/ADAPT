from pathlib import Path
import json
import time
import ollama

class QueryAnalyser:
    def __init__(self,service_desc_path,model_name):
        print(f"[QueryAnalyser] Loading service descriptions from {service_desc_path}...")
        self.service_desc_path = service_desc_path
        self.model_name = model_name
        self.services = self._get_service_descriptions() 
        print(f"[QueryAnalyser] Loaded {len(self.services)} services: {list(self.services.keys())}")
        print(f"[QueryAnalyser] Ready to route using model: {model_name}")

    def _get_service_descriptions(self):
        # change service_desc_path from string to path
        path = Path(self.service_desc_path)
        if not path.exists():
            raise FileNotFoundError(f"Service description file '{self.service_desc_path}' not found.")
        with open(path, "r") as f:
            data = json.load(f)
            # print(f"[LOADED SERVICES] {data}")
            return data

    def _get_system_prompt(self):
        return (
            "You are a query router for a multi-service assistant system. "
            "On each turn you will be given a list of available services with short descriptions, followed by a user query. "
            "Decide which single service is best suited to handle that query, using only the descriptions provided for this turn "
            "Never assume a service exists beyond what is listed.\n\n"
            "Rules:\n"
            "- Match the query against the core intent of each service description, not just surface keywords.\n"
            "- If more than one service seems plausible, choose the one whose description is the more specific and direct match.\n"
            "- Respond with only the service name in the exact form - 'service $name', where 'name' is the exact name given for that service.\n"
            "Do not include any explanation or additional text."
        )

    def _build_user_message(self, query):
        active_services = [f"- {name}: {desc}" for name, desc in self.services.items()]
        active_services_joined = "\n".join(active_services)
        message= (
            f"Services: {active_services_joined}\n\n"
            f"Query: \"{query}\"\n\n"
            "Which service should handle this query?"
            )
        # print(f"[USER MESSAGE SENT TO MODEL]\n{message}\n")
        return message


    def route(self, query):
        start = time.perf_counter()
        response = ollama.chat(
            model = self.model_name,
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": self._build_user_message(query)},])
        latency_ms = (time.perf_counter() - start) * 1000
        predicted_label = response["message"]["content"]
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)

        return {
            "query": query,
            "predicted_label": predicted_label,
            "model_name": self.model_name, 
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

# if __name__ == "__main__":
#     analyser = QueryAnalyser(service_desc_path="../../data/services_desc.json", model="qwen2.5:7b")
#     result = analyser.route("Find the index of <p> in S_5.")
#     print(result)


