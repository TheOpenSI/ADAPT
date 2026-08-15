import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")


from dataset_builder.mmlu_builder_base import MMLUBenchmarkBuilderBase
import pandas as pd
from config import SERVICE_SETS

class MMLUTestBuilder(MMLUBenchmarkBuilderBase):
    def __init__(
        self,
        split="test",
        per_general_subject_cap=50,
        random_state=42
    ):
        super().__init__(split)

        self.per_general_subject_cap = per_general_subject_cap
        self.random_state = random_state


    def _cap_general(self, df):
        """
        Sample a fixed number of questions from each general-source subject
        while keeping all specialist-service questions.
        """
        print("\n[4] Sampling general subjects...")

        general_df = df[
            df["expected_service"] == "general"
        ]

        specialist_df = df[
            df["expected_service"] != "general"
        ]

        sampled_general = (
            general_df
            .groupby("subject", group_keys=False)
            .sample(
                n=self.per_general_subject_cap,
                random_state=self.random_state
            )
        )

        print("[4] Sampled general subject counts:")
        print(sampled_general["subject"].value_counts())

        final_data = pd.concat(
            [specialist_df, sampled_general],
            ignore_index=True
        )

        print("[4] Expected service counts after sampling:")
        print(final_data["expected_service"].value_counts())

        return final_data


    def _select_columns(self, df):
        """
        Keep only the columns required for final routing evaluation.
        """
        return df[
            ["question", "expected_service"]
        ].copy()

    def _save_files(self, df, service_id):
        """
        Save the final test dataset as CSV and JSON.
        """

        base_path = f"../dataset/test_service_{service_id}"

        df.to_csv(
            f"{base_path}.csv",
            index=False
            )

        df.to_json(
            f"{base_path}.json",
            orient="records",
            indent=2
            )

        print(f"[5] Saved: {base_path}.csv")
        print(f"[5] Saved: {base_path}.json")
    

    def _build_dataset(self, subjects):
        print("\n" + "=" * 60)
        print("BUILDING TEST DATASET")
        print("=" * 60)

        filtered_data = self._filter_subjects(subjects)

        converted_data = self._convert_subject(
            filtered_data,
            subjects
        )

        sampled_data = self._cap_general(
            converted_data
        )

        routing_data = self._select_columns(
            sampled_data
        )

        return routing_data

    def build_and_save(self, service_id, subjects):
        """
        Build and save the final test dataset for one service set.
        """

        routing_data = self._build_dataset(subjects)

        self._save_files(routing_data,service_id)

        print(f"[DONE] test_service_{service_id}")
        print(f"[DONE] Total rows: {len(routing_data)}")
        print(routing_data["expected_service"].value_counts())

        return routing_data


if __name__ == "__main__":

    builder = MMLUTestBuilder()

    for service_id, subjects in SERVICE_SETS.items():
        builder.build_and_save(
            service_id,
            subjects
        )


    