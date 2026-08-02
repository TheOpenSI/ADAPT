from pathlib import Path

import pandas as pd
import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from services.QueryAnalyser import QueryAnalyser


class Evaluator:
    def __init__(self, dataset_path, router, results_dir="../results"):
        self.dataset_path = dataset_path
        self.df = pd.read_csv(dataset_path)
        self.router = router

        dataset_name = Path(self.dataset_path).stem
        self.results_dir = Path(results_dir) / dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _build_run_name(self):
        router_type = self.router.__class__.__name__   
        model_name = self.router.model_name
    
        return f"{router_type}_{model_name}"

    
    def _get_result(self):
        result = []

        for i in range(len(self.df[:3])):
            prompt = self.df.iloc[i]["prompt"]
            expected_service = self.df.iloc[i]["expected_service"]
            route = self.router.route(prompt)
            route["expected_service"] = expected_service 
            route["is_correct"] = (route["predicted_label"] == route["expected_service"])
            result.append(route)

        return result


    def _analyze_result(self,result):
        result_df = pd.DataFrame(result)
        
        return(result_df.head())


    def _save(self, result_df, run_name):
        json_path = self.results_dir / f"{run_name}.json"
        result_df.to_json(json_path, orient="records", indent=2)

        csv_path = self.results_dir / f"{run_name}.csv"
        result_df.to_csv(csv_path, index=False)

        print(f"saved: {run_name},{json_path}")
        print(f"saved: {run_name},{csv_path}")


    def run_test(self):
        result = self._get_result()
        summary = self._analyze_result(result)
        run_name = self._build_run_name()
        self._save(summary, run_name)


if __name__ == "__main__":
    router = QueryAnalyser(service_desc_path="../data/services_desc.json", model_name="qwen2.5:7b")

    a = Evaluator("../data/dataset_3_services.csv", router)
    summary = a.run_test()
    print(summary)
     