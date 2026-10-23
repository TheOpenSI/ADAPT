from pathlib import Path
import pandas as pd
import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from codecarbon import EmissionsTracker


class BenchmarkRunner:
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
        routing_method = self.router.routing_method

        return f"{routing_method}_{model_name}"


    def _get_result(self, run_name):

        total = len(self.df)

        correct_count = 0
        wrong_count = 0
        error_count = 0

        result = []

        # One tracker for the whole run, one task per query
        tracker = EmissionsTracker(
            project_name=run_name,
            save_to_file=False,
            allow_multiple_runs=True,
            tracking_mode="process",
            log_level="error",
            measure_power_secs=1,
        )
        tracker.start()

        for i in range(total):

            prompt = self.df.iloc[i]["question"]
            expected_service = self.df.iloc[i]["expected_service"]

            print(f"\n[{i+1}/{total}] Question: {prompt}")

            tracker.start_task(f"query_{i}")
            try:
                route = self.router.route(prompt)

                route["expected_service"] = expected_service
                route["is_correct"] = (route["predicted_label"] == route["expected_service"])
                route["error"] = None

                if route["is_correct"]:
                    correct_count += 1
                    status = "✓ CORRECT"

                else:
                    wrong_count += 1
                    status = "✗ WRONG"

                print(
                    f"→ predicted={route['predicted_label']} | "
                    f"expected={expected_service} | "
                    f"{status} | "
                    f"Correct: {correct_count} | "
                    f"Wrong: {wrong_count} | "
                    f"{route['latency_ms']:.1f}ms"
                )

            except Exception as e:
                error_count += 1
                route = {
                    "routing_method": getattr(self.router, "routing_method", None),
                    "query": prompt,
                    "predicted_label": None,
                    "model_name": getattr(self.router, "model_name", None),
                    "latency_ms": None,
                    "expected_service": expected_service,
                    "is_correct": False,
                    "error": str(e),
                }
                print(f"→ ERROR | Errors: {error_count} | {e}")

            emissions_data = tracker.stop_task()
            route["energy_kwh"] = emissions_data.energy_consumed if emissions_data else None
            route["co2_kg"] = emissions_data.emissions if emissions_data else None

            result.append(route)

        tracker.stop()

        print("\n=== ROUTING RESULTS ===")
        print(f"Correct: {correct_count}")
        print(f"Wrong:   {wrong_count}")
        print(f"Errors:  {error_count}")
        print(f"Total:   {total}")

        return result

    def _save_result(self, result_df, run_name):
        json_path = self.results_dir / f"{run_name}.json"
        result_df.to_json(json_path, orient="records", indent=2)

        csv_path = self.results_dir / f"{run_name}.csv"
        result_df.to_csv(csv_path, index=False)

        print(f"saved: {json_path}")
        print(f"saved: {csv_path}")


    def run_test(self):
        print(f"\n{'='*60}")
        print(f"Starting benchmark: {self.router.__class__.__name__} on {self.dataset_path}")
        print(f"Total rows to process: {len(self.df)}")
        print(f"{'='*60}\n")

        run_name = self._build_run_name()

        result = self._get_result(run_name)

        total_service_count = len(self.router.services)

        result_df = pd.DataFrame(result)
        result_df["total_service_count"] = total_service_count

        self._save_result(result_df,run_name)

        print(f"DONE")
        print(f"{'='*60}\n")

        return result_df