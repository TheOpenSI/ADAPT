import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from dataset_builder.config import SERVICE_SETS
import ollama
from pathlib import Path
import pandas as pd


class RoutingTester:
    """
    Routing evaluation: run each profiling question through an LLM router
    and compare its predicted service against the expected service (without description).
    """

    def __init__(self, services, model="qwen2.5:7b",results_dir = "../data",):
        self.services = services
        self.model = model
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)



    def _build_system_prompt(self):
        services_text = "\n".join(f"- {service}" for service in self.services)
        return f"""You are a query-routing classifier.

Available services:
{services_text}

Do not create, rename, or infer any other service.
Your answer must exactly match one of the service names above.

Use general only when the query does not belong to any listed specialist service.

Select exactly one service.
Return only the service name."""


    def _build_user_prompt(self, question, choices):
        choices_text = "\n".join(
            f"{chr(65 + i)}. {choice}"
            for i, choice in enumerate(choices)
            )

        return f"""Question:
{question}

Choices:
{choices_text}"""


    def _route(self, question, choices):
        response = ollama.chat(
            model= self.model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_user_prompt(question, choices)},
            ],
            options={"temperature": 0},
        )

        return response["message"]["content"].strip()


    def _route_dataset(self, df):
        predictions = []
        total = len(df)

        for i in range(total):
            question = df.iloc[i]["question"]
            choices = df.iloc[i]["choices"]
            expected_service = df.iloc[i]["expected_service"]
            print(f"\n[{i + 1}/{total}] Routing question")
            print(f"Question: {question}")
            print(f"Expected: {expected_service}")


            prediction = self._route(question,choices)

            if prediction not in self.services:
                print(f"[INVALID] First prediction: {prediction}")
                print("          Retrying...")

                prediction = self._retry_route(question,choices)
                if prediction not in self.services:
                    print(f"[INVALID AGAIN] {prediction}")
                    prediction = "__invalid__"

            print(f"Predicted: {prediction}")
            print(f"Correct: {prediction == expected_service}")

            predictions.append(prediction)

        df = df.copy()
        df["predicted_service"] = predictions
        df["correct"] = df["predicted_service"] == df["expected_service"]

        return df

    def _retry_route(self, question, choices):
        services_text = "\n".join(
            f"- {service}"
            for service in self.services
            )

        response = ollama.chat(
            model=self.model,
            messages=[
            {
                "role": "system",
                "content": f"""You are a query-routing classifier.

You MUST select exactly one service from this list:

{services_text}

Do not create, rename, infer, or output any other label.
Return only the exact service name."""
            },
            {
                "role": "user",
                "content": self._build_user_prompt(
                    question,
                    choices)
            }
        ],
        options={"temperature": 0},
    )

        return response["message"]["content"].strip()



    def _save_files(self, df):
        path = self.results_dir / f"routing_services_result"

        df.to_csv(f"{path}.csv", index=False)
        df.to_json(f"{path}.json", orient="records", indent=2)

        return path

    def run_test(self, df):
        print(f"[1] Routing {len(df)} questions through {self.model}")

        df = self._route_dataset(df)
        print(f"[2] Predictions generated")
        print(f"    {df['correct'].value_counts().to_dict()}")

        save_path = self._save_files(df)
        print(f"[3] Routing results saved to {save_path}.csv / {save_path}.json")

        return df


# if __name__ == "__main__":
#     services = SERVICE_SETS[3] + ["general"]
#     routing_tester = RoutingTester(services=services)
#     df = pd.read_csv('../profiling_dataset_services3.csv')
#     results = routing_tester.run_test(df)