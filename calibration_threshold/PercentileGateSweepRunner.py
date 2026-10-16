import os
import numpy as np
import pandas as pd

from data_collector.BenchmarkRunner import BenchmarkRunner


class PercentileGateSweepRunner:
    """Percentile sweep where the ROUTER applies the gap gate itself.

    For each (model, k) the model is loaded once. For each percentile p the router's gate is set to
        gate_mode="gap", gap_c0=c0(p), gap_c1=c1(p)   ->  accept if  max - median > c0 + c1*ln k
    and the validation file is routed. The router's own predicted_label is the result.
    """

    def __init__(self, router_cls, model_name, conditions, data_dir, results_dir,
                 constants, verbose=False):
        self.router_cls = router_cls
        self.model_name = model_name
        self.conditions = conditions
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.constants = constants                    # {percentile: (c0, c1)}
        self.percentiles = sorted(constants)
        self.verbose = verbose
        os.makedirs(self.results_dir, exist_ok=True)

    def _calculate_metrics(self, result_df):
        accuracy = result_df["is_correct"].mean()
        fallback_rate = (result_df["predicted_label"] == "general").mean()

        specialist_df = result_df[result_df["expected_service"] != "general"]
        general_df = result_df[result_df["expected_service"] == "general"]

        specialist_recall = (
            (specialist_df["predicted_label"] == specialist_df["expected_service"]).mean()
            if len(specialist_df) > 0 else np.nan
        )
        general_recall = (
            (general_df["predicted_label"] == "general").mean()
            if len(general_df) > 0 else np.nan
        )
        return {
            "n_general": len(general_df),
            "n_specialist": len(specialist_df),
            "accuracy": round(float(accuracy), 4),
            "fallback_rate": round(float(fallback_rate), 4),
            "specialist_recall": round(float(specialist_recall), 4),
            "general_recall": round(float(general_recall), 4),
            "balanced_accuracy": round(float(np.nanmean([specialist_recall, general_recall])), 4),
        }

    def run(self):
        summary = []

        for k in self.conditions:
            service_desc_path = os.path.join(
                self.data_dir, "dataset", f"service_descriptions_{k}.json")
            validation_path = os.path.abspath(os.path.join(
                self.data_dir, "..", "calibration_threshold", "dataset",
                f"validation_eval_service_{k}.csv"))

            print("\n" + "=" * 80)
            print(f"MODEL={self.model_name} | k={k}")
            print(f"Validation dataset: {validation_path}")
            print("=" * 80)

            first_c0, first_c1 = self.constants[self.percentiles[0]]
            router = self.router_cls(
                service_desc_path=service_desc_path,
                model_name=self.model_name,
                required_strength_percentage=0,
                apply_threshold=True,
                gate_mode="gap",
                gap_c0=first_c0,
                gap_c1=first_c1,
                verbose=self.verbose,
            )                                         # model is loaded ONCE per (model, k)

            for p in self.percentiles:
                c0, c1 = self.constants[p]
                expected_cutoff = c0 + c1 * np.log(k)

                # switch the router to this percentile's constants
                router.apply_threshold = True
                router.gate_mode = "gap"
                router.gap_c0, router.gap_c1 = c0, c1

                print(f"\n--- p={p} | cutoff T({k}) = {expected_cutoff:.6g} ---")
                benchmark = BenchmarkRunner(
                    dataset_path=validation_path,
                    router=router,
                    results_dir=os.path.join(self.results_dir, f"p{p}"),   # one folder per percentile
                )
                result_df = benchmark.run_test()
                result_df = result_df[result_df["predicted_label"].notna()]   # drop crashed rows

                # safety: the router must have used exactly the intended cutoff
                if "gap_cutoff" in result_df.columns:
                    used = result_df["gap_cutoff"].astype(float)
                    if not np.allclose(used, expected_cutoff, rtol=1e-6, atol=0):
                        raise ValueError(
                            f"Router used cutoff {used.iloc[0]:.6g} but expected {expected_cutoff:.6g} "
                            f"(k={k}). Does service_descriptions_{k}.json contain a 'general' entry?")

                metrics = self._calculate_metrics(result_df)
                summary.append({
                    "router": router.routing_method,
                    "model_name": self.model_name,
                    "k": k,
                    "percentile": p,
                    "c0": c0,
                    "c1": c1,
                    "cutoff": expected_cutoff,
                    **metrics,
                })
                print(f"  p={p}: {metrics}")

            try:
                if hasattr(router, "unload_model"):
                    router.unload_model()
            except Exception as e:
                print(f"Unload warning: {e}")

        summary_df = pd.DataFrame(summary).sort_values(["k", "percentile"]).reset_index(drop=True)
        model_file = self.model_name.replace("/", "_").replace(":", "_")
        output_path = os.path.join(self.results_dir, f"gate_sweep_{model_file}.csv")
        summary_df.to_csv(output_path, index=False)

        print("\n" + "=" * 80)
        print(f"PERCENTILE SWEEP (router-applied gap gate): {self.model_name}")
        print("=" * 80)
        print(summary_df[["k", "percentile", "cutoff", "general_recall",
                          "specialist_recall", "balanced_accuracy"]])
        print(f"\nSaved: {output_path}")
        return summary_df