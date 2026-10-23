from pathlib import Path
import json
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


    def _get_result(self):

        total = len(self.df)

        correct_count = 0
        wrong_count = 0
        error_count = 0

        result = []

        for i in range(total):

            prompt = self.df.iloc[i]["question"]
            expected_service = self.df.iloc[i]["expected_service"]
            query_id = (
                self.df.iloc[i]["query_id"]
                if "query_id" in self.df.columns
                else i
            )

            print(f"\n[{i+1}/{total}] Question: {prompt}")

            try:
                route = self.router.route(prompt)

                route["query_id"] = query_id
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
                        "query_id" : query_id,
                        "routing_method": getattr(self.router, "routing_method", None),
                        "query": prompt,
                        "predicted_label": None,
                        "model_name": getattr(self.router, "model_name", None),
                        "latency_ms": None,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "raw_response": None,
                        "expected_service": expected_service,
                        "is_correct": False,
                        "error": str(e),
                    }
                    print(
                        f"→ ERROR | "
                        f"Correct: {correct_count} | "
                        f"Wrong: {wrong_count} | "
                        f"Errors: {error_count} | "
                        f"{e}"
                    )

            result.append(route)

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


    def _save_efficiency(self, efficiency_result, run_name):
        path = (self.results_dir/ f"{run_name}_efficiency.json")

        with open(path, "w") as f:
            json.dump(efficiency_result, f, indent=2)
        print(f"saved: {path}")


    def run_test(self):
        print(f"\n{'='*60}")
        print(f"Starting benchmark: {self.router.__class__.__name__} on {self.dataset_path}")
        print(f"Total rows to process: {len(self.df)}")
        print(f"{'='*60}\n")

        run_name = self._build_run_name()
        
        tracker = EmissionsTracker(project_name=run_name, output_dir=str(self.results_dir),tracking_mode="process")

        tracker.start()

        result = self._get_result()
        
        tracker.stop()

        specialist_service_count = len([
             service
             for service in self.router.services
             if service != "general"
            ])
        total_service_count = len(self.router.services)

        # Query-level results
        result_df = pd.DataFrame(result)
        result_df["specialist_service_count"] = specialist_service_count
        result_df["total_service_count"] = total_service_count

        self._save_result(result_df,run_name)

        # Run-level CodeCarbon results
        total_queries = len(result)

        total_co2_kg = (tracker.final_emissions_data.emissions)
        total_energy_kwh = (tracker.final_emissions_data.energy_consumed)

        co2_per_query_kg = (total_co2_kg / total_queries if total_queries > 0 else None)
        energy_per_query_kwh = (total_energy_kwh / total_queries if total_queries > 0 else None)

        efficiency_result = {
            "routing_method" : self.router.routing_method,
            "model_name" : self.router.model_name,
            "specialist_service_count" : specialist_service_count,
            "total_service_count": total_service_count,
            "total_queries" : total_queries,
            "total_energy_kwh" : total_energy_kwh,
            "energy_per_query_kwh" : energy_per_query_kwh,
            "total_co2_kg" : total_co2_kg,
            "co2_per_query_kg": co2_per_query_kg  
        }

        self._save_efficiency(efficiency_result,run_name)

        print("\n=== ENERGY / CARBON ===")
        print("Total CO2 (kg):", total_co2_kg)
        print("Total energy (kWh):", total_energy_kwh)

        print(f"DONE")
        print(f"{'='*60}\n")

        return result_df