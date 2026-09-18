import os
import pandas as pd

from data_collector.BenchmarkRunner import BenchmarkRunner


class CalibrationRunner:

    def __init__(
        self,
        router_cls,
        model_name,
        thresholds,
        conditions,
        data_dir,
        results_dir,
        verbose=False,
    ):
        self.router_cls = router_cls
        self.model_name = model_name
        self.thresholds = thresholds
        self.conditions = conditions
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.verbose = verbose

        os.makedirs(
            self.results_dir,
            exist_ok=True
        )

    def _calculate_metrics(self, result_df):

        accuracy = result_df["is_correct"].mean()

        fallback_rate = (
            result_df["predicted_label"] == "general"
        ).mean()

        specialist_df = result_df[
            result_df["expected_service"] != "general"
        ]

        general_df = result_df[
            result_df["expected_service"] == "general"
        ]

        if len(specialist_df) > 0:
            specialist_recall = (
                specialist_df["predicted_label"]
                == specialist_df["expected_service"]
            ).mean()
        else:
            specialist_recall = 0

        if len(general_df) > 0:
            general_recall = (
                general_df["predicted_label"]
                == "general"
            ).mean()
        else:
            general_recall = 0

        return {
            "accuracy": round(float(accuracy), 4),
            "fallback_rate": round(float(fallback_rate), 4),
            "specialist_recall": round(
                float(specialist_recall),
                4
            ),
            "general_recall": round(
                float(general_recall),
                4
            ),
        }


    def run(self):

        summary = []

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
                f"validation_service_{k}.csv"
            )

            validation_path = os.path.abspath(
                validation_path
            )

            print("\n" + "=" * 80)
            print(f"CONDITION k={k}")
            print(f"Validation Dataset: {validation_path}")
            print("=" * 80)

            for threshold in self.thresholds:

                print("\n" + "-" * 80)
                print(
                    f"MODEL={self.model_name} | "
                    f"THRESHOLD={threshold}"
                )
                print("-" * 80)

                router = self.router_cls(
                    service_desc_path=service_desc_path,
                    model_name=self.model_name,
                    confidence_c=threshold,
                    verbose=self.verbose,
                )

                benchmark = BenchmarkRunner(
                    dataset_path=validation_path,
                    router=router,
                    results_dir=self.results_dir,
                )

                result_df = benchmark.run_test()

                metrics = self._calculate_metrics(
                    result_df
                )

                summary.append({
                    "router": router.routing_method,
                    "model_name": self.model_name,
                    "k": k,
                    "threshold": threshold,
                    **metrics
                })

                print("\nMetrics:")
                print(metrics)

                try:
                    if hasattr(router, "unload_model"):
                        router.unload_model()
                except Exception as e:
                    print(
                        f"Unload warning: {e}"
                    )

        summary_df = pd.DataFrame(summary)

        output_path = os.path.join(
            self.results_dir,
            "calibration_summary.csv"
        )

        summary_df.to_csv(
            output_path,
            index=False
        )

        print("\n" + "=" * 80)
        print("CALIBRATION SUMMARY")
        print("=" * 80)
        print(summary_df)

        print(
            f"\nSaved calibration summary:"
        )
        print(output_path)

        return summary_df