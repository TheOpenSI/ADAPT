import ast
import os

import numpy as np
import pandas as pd

from data_collector.BenchmarkRunner import BenchmarkRunner


class TopNCalibrationRunner:
    """
    The result is the top_n table:

        top_n | Off-topic that slipped through (of N) | Specialists let through (of M)

    It runs the router once per K on the validation set (gate off, so only the scores matter),
    then tries every top_n from the saved scores.
    """

    def __init__(
        self,
        router_cls,
        model_name,
        conditions,
        data_dir,
        results_dir,
        reference_queries,
        percentile=10,
        verbose=False,
    ):
        self.router_cls = router_cls
        self.model_name = model_name
        self.conditions = conditions
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.reference_queries = reference_queries
        self.percentile = percentile
        self.verbose = verbose

        os.makedirs(
            self.results_dir,
            exist_ok=True
        )

    def _lead(self, scores, top_n):
        leads = []

        for query_scores in scores:                              # one row = one query's score for every specialist
            best_first = sorted(query_scores, reverse=True)      # highest score first
            top_scores = best_first[:top_n]                      # keep only the best top_n scores

            best = top_scores[0]                                 # the winner's score
            others = top_scores[1:]                              # the runners-up among the kept scores

            leads.append(best - sum(others) / len(others))       # winner minus the average of the runners-up

        return np.array(leads)

    def _top_n_table(self, result_df, reference_scores, names):

        scores = [s if isinstance(s, dict) else ast.literal_eval(s) for s in result_df["scores"]]
        test_scores = np.array([[s[n] for n in names] for s in scores])

        is_specialist = (result_df["expected_service"] != "general").values
        n_specialist = int(is_specialist.sum())
        n_general = int((~is_specialist).sum())

        rows = []

        for top_n in range(2, len(names) + 1):

            cutoff = np.quantile(
                self._lead(reference_scores, top_n),
                1 - self.percentile / 100
            )

            passed = self._lead(test_scores, top_n) > cutoff      # True = the shield lets the query through

            rows.append({
                "top_n": top_n,
                f"Off-topic that slipped through (of {n_general})": int(passed[~is_specialist].sum()),
                f"Specialists let through (of {n_specialist})": int(passed[is_specialist].sum()),
            })

        return pd.DataFrame(rows)

    def run(self):

        all_tables = []

        for k in self.conditions:

            service_desc_path = os.path.join(
                self.data_dir,
                "dataset",
                f"service_descriptions_{k}.json"
            )

            validation_path = os.path.join(
                self.data_dir,
                "..",
                "calibration_threshold",
                "dataset",
                f"validation_eval_service_{k}.csv"
            )

            validation_path = os.path.abspath(
                validation_path
            )

            print("\n" + "=" * 80)
            print(f"CONDITION k={k}")
            print(f"Validation Dataset: {validation_path}")
            print("=" * 80)

            router = self.router_cls(
                service_desc_path=service_desc_path,
                model_name=self.model_name,
                reference_queries=self.reference_queries,
                percentile=self.percentile,          # only a placeholder here, the table below tries all top_n
                gate_mode="off",                     # no gate, we only want the scores
                verbose=self.verbose,
            )

            benchmark = BenchmarkRunner(
                dataset_path=validation_path,
                router=router,
                results_dir=self.results_dir,
            )

            result_df = benchmark.run_test()

            # the reference scores the router saved (rows = off-topic reference queries, columns = specialists)
            names = list(router.services.keys())
            reference_scores = pd.read_csv(router.gate.cache_file, index_col=0)[names].values

            table = self._top_n_table(result_df, reference_scores, names)

            print("\n" + "=" * 80)
            print(f"TOP_N TABLE  |  K={k}  |  percentile {self.percentile}")
            print("=" * 80)
            print(table.to_string(index=False))

            table_path = os.path.join(self.results_dir, f"top_n_table_k{k}.csv")
            table.to_csv(table_path, index=False)
            print(f"\nSaved: {table_path}")

            all_tables.append(table.assign(k=k))

            try:
                if hasattr(router, "unload_model"):
                    router.unload_model()
            except Exception as e:
                print(f"Unload warning: {e}")

        return pd.concat(all_tables, ignore_index=True)