from pathlib import Path
import json
import pandas as pd
import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from services.ColBERTRouter import ColBERTRouter
from services.CrossEncoderRouter import CrossEncoderRouter
from services.SemanticRouter import SemanticRouter
from services.QueryAnalyser import QueryAnalyser
from sklearn.metrics import classification_report, confusion_matrix
from codecarbon import EmissionsTracker


class Evaluator:
    def __init__(self, dataset_path, router, results_dir="../results"):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.router = router

        dataset_name = Path(self.dataset_path).stem
        self.results_dir = Path(results_dir) / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_run_name(self): 
        model_name = self.router.model_name
        model_name = self.router.model_name.replace("/", "-") 
    
        return f"{model_name}"

    
    def _get_result(self):
        result = []
        total = len(self.df)

        for i in range(total):
            prompt = self.df.iloc[i]["question"]
            expected_service = self.df.iloc[i]["expected_service"]
            route = self.router.route(prompt)
            route["expected_service"] = expected_service 
            route["is_correct"] = (route["predicted_label"] == route["expected_service"])
            result.append(route)

            print(f"[{i+1}/{total}] predicted={route['predicted_label']} expected={expected_service} "
                  f"{'✓' if route['is_correct'] else '✗'} latency={route['latency_ms']:.1f}ms")

        return result


    def _analyze_result(self,result):
        result_df = pd.DataFrame(result)
        correct_count = len(result_df[result_df["is_correct"] == True])
        total = len(result_df)
        accuracy = correct_count / total

        report = classification_report(
        y_true=result_df["expected_service"],
        y_pred=result_df["predicted_label"],
        output_dict=True,
        zero_division=0,   # avoids warnings/errors if 0/0
        )

        labels = sorted(result_df["expected_service"].unique())
        cm = confusion_matrix(
        result_df["expected_service"],
        result_df["predicted_label"],
        labels=labels,
        )

        summary = {
            "correct_count": correct_count,
            "total": total,
            "accuracy": accuracy,
            "avg_latency_ms": result_df["latency_ms"].mean(),
            "avg_tokens": result_df["total_tokens"].mean(),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": labels,
        }

        return(result_df, summary)


    def _save_result(self, result_df, run_name):
        json_path = self.results_dir / f"{run_name}.json"
        result_df.to_json(json_path, orient="records", indent=2)

        csv_path = self.results_dir / f"{run_name}.csv"
        result_df.to_csv(csv_path, index=False)

        print(f"saved: {json_path}")
        print(f"saved: {csv_path}")


    def _save_summary(self, summary,run_name):
        json_path = self.results_dir / f"{run_name}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"saved summary: {json_path}")

        
    def run_test(self):
        print(f"\n{'='*60}")
        print(f"Starting evaluation: {self.router.__class__.__name__} on {self.dataset_path}")
        print(f"Total rows to process: {len(self.df)}")
        print(f"{'='*60}\n")

        run_name = self._build_run_name()
         # start carbon tracking for the WHOLE run
        tracker = EmissionsTracker(project_name=run_name, output_dir=str(self.results_dir))
        tracker.start()

        result = self._get_result()

        total_emissions = tracker.stop()

        result_df, summary = self._analyze_result(result)

        print("=== TOTALS ===")
        print("Total carbon emissions (kg CO2):", total_emissions)
        print("Total energy consumed (kWh):", tracker.final_emissions_data.energy_consumed)
        print("\n=== BREAKDOWN ===")
        print(tracker.final_emissions_data)

        self._save_result(result_df, run_name)
        self._save_summary(summary, run_name) 

        print(f"\n{'='*60}")
        print(f"DONE — accuracy: {summary['accuracy']:.3f}")
        print(f"{'='*60}\n")

        return summary


if __name__ == "__main__":
    # router = QueryAnalyser(
    #     service_desc_path="../data/services_desc.json",
    #     model_name="qwen2.5:32b",
    # )

    # router = QueryAnalyser(service_desc_path="../data/services_desc.json", model_name="qwen2.5:7b")

    # router = CrossEncoderRouter(service_desc_path="../data/services_desc.json", model_name="cross-encoder/ms-marco-MiniLM-L6-v2")

    # router = QueryAnalyser(
    #     service_desc_path="../data/services_desc.json",  
    #     model_name="qwen2.5:32b",
    # )
    
    router = SemanticRouter(
        service_desc_path="../data/services_desc.json",
        model_name="Qwen/Qwen3-Embedding-8B",

        )


    a = Evaluator("../data/dataset/service_10.csv", router)
    summary = a.run_test()
    print(summary)