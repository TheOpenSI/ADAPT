from pathlib import Path
import json
import pandas as pd
import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from sklearn.metrics import classification_report, confusion_matrix
from codecarbon import EmissionsTracker


class Evaluator:
    def __init__(self, dataset_path, router, results_dir="../results", resume=True):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.router = router
        self.resume = resume

        dataset_name = Path(self.dataset_path).stem
        self.results_dir = Path(results_dir) / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_run_name(self): 
        model_name = self.router.model_name
        model_name = self.router.model_name.replace("/", "-") 
        routing_method = self.router.routing_method

        return f"{routing_method}_{model_name}"

    
    def _load_existing_results(self, jsonl_path):
        """Reads a previous (possibly crashed) run's jsonl file, if one exists,
        and returns the rows already completed so _get_result can skip them."""
        if not jsonl_path.exists():
            return []

        existing = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.append(json.loads(line))

        print(f"[resume] Found existing run with {len(existing)} completed queries — resuming from row {len(existing)}.")
        return existing

    def _get_result(self, tracker):
        """
        Loops through every query, calls router.route() on each, and writes
        each row to JSONL IMMEDIATELY as it's produced — so if the run
        crashes partway through, everything up to that point is already saved
        on disk instead of lost. 

        If a jsonl from a previous (crashed) attempt at this same run exists,
        already-completed queries are skipped and new rows are appended —
        so retrying after a crash resumes instead of starting over.

        A single query failing is caught and logged as an error row instead
        of stopping the whole run.
        """
        total = len(self.df)
        error_count = 0

        run_name = self._build_run_name()
        jsonl_path = self.results_dir / f"{run_name}.jsonl"

        if self.resume:
            result = self._load_existing_results(jsonl_path)
        else:
            if jsonl_path.exists():
                print(f"[resume=False] Discarding existing {jsonl_path.name} — starting fresh.")
                jsonl_path.unlink()
            result = []

        start_index = len(result)

        if start_index >= total:
            print(f"[resume] All {total} queries already completed — nothing to do.")
            return result

        with open(jsonl_path, "a") as jsonl_f:   
            prev_co2_kg = 0.0

            for i in range(start_index, total):
                prompt = self.df.iloc[i]["question"]
                expected_service = self.df.iloc[i]["expected_service"]

                try:
                    route = self.router.route(prompt)

                    current_co2_kg = tracker.flush()
                    route["co2_kg"] = current_co2_kg - prev_co2_kg
                    prev_co2_kg = current_co2_kg

                    route["expected_service"] = expected_service
                    route["is_correct"] = (route["predicted_label"] == route["expected_service"])
                    route["error"] = None

                    print(f"[{i+1}/{total}] predicted={route['predicted_label']} expected={expected_service} "
                          f"{'✓' if route['is_correct'] else '✗'} latency={route['latency_ms']:.1f}ms "
                          f"co2={route['co2_kg']:.8f}kg")

                except Exception as e:
                    error_count += 1
                    route = {
                        "routing_method": getattr(self.router, "routing_method", None),
                        "query": prompt,
                        "predicted_label": None,
                        "model_name": getattr(self.router, "model_name", None),
                        "latency_ms": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "gpu_memory_gb": None,
                        "co2_kg": None,
                        "expected_service": expected_service,
                        "is_correct": False,
                        "error": str(e),
                    }
                    print(f"[{i+1}/{total}] ERROR on query: {e}")

                result.append(route)

                jsonl_f.write(json.dumps(route) + "\n")
                jsonl_f.flush()

        if error_count > 0:
            print(f"\nWARNING: {error_count}/{total} queries failed — see 'error' column in output files")

        return result


    def _analyze_result(self, result):
        result_df = pd.DataFrame(result)

        valid_df = result_df[result_df["predicted_label"].notna()]
        failed_count = len(result_df) - len(valid_df)
        if failed_count > 0:
            print(f"Excluding {failed_count} failed queries from classification metrics")

        correct_count = len(valid_df[valid_df["is_correct"] == True])
        total = len(result_df)
        accuracy = correct_count / total if total > 0 else 0.0

        report = classification_report(
            y_true=valid_df["expected_service"],
            y_pred=valid_df["predicted_label"],
            output_dict=True,
            zero_division=0,   # avoids warnings/errors if 0/0
        )

        labels = sorted(valid_df["expected_service"].unique())

        cm = confusion_matrix(
            valid_df["expected_service"],
            valid_df["predicted_label"],
            labels=labels,
        )

        macro_precision = report["macro avg"]["precision"]
        macro_recall = report["macro avg"]["recall"]
        macro_f1 = report["macro avg"]["f1-score"]

        specialist_service_count = len([
            service
            for service in self.router.services.keys()
            if service != "general"
        ])

        summary = {
            # Experiment information
            "routing_method": self.router.routing_method,
            "model_name": self.router.model_name,
            "specialist_service_count": specialist_service_count,
            "total_routing_labels": len(self.router.services),

            # RQ1 metrics
            "correct_count": correct_count,
            "total": total,
            "failed_count": failed_count,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,

            # RQ3 metrics
            "avg_latency_ms": valid_df["latency_ms"].mean(),
            "avg_tokens": valid_df["total_tokens"].mean(),

            # RQ4
            "avg_gpu_memory_gb": valid_df["gpu_memory_gb"].mean() if "gpu_memory_gb" in valid_df else None,
            "avg_co2_kg": valid_df["co2_kg"].mean() if "co2_kg" in valid_df else None,

            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": labels,
        }

        return (result_df, summary)


    def _save_result(self, result_df, run_name):
        json_path = self.results_dir / f"{run_name}.json"
        result_df.to_json(json_path, orient="records", indent=2)

        csv_path = self.results_dir / f"{run_name}.csv"
        result_df.to_csv(csv_path, index=False)

        print(f"saved: {json_path}")
        print(f"saved: {csv_path}")


    def _save_summary(self, summary, run_name):
        json_path = self.results_dir / f"{run_name}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"saved summary: {json_path}")

        
    def run_test(self):
        print(f"\n{'='*60}")
        print(f"Starting evaluation: {self.router.__class__.__name__} on {self.dataset_path}")
        print(f"Total rows to process: {len(self.df)}")
        print(f"{'='*60}\n")

        run_name = self._build_run_name()
        
        tracker = EmissionsTracker(project_name=run_name, output_dir=str(self.results_dir))
        tracker.start()

        result = self._get_result(tracker)

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