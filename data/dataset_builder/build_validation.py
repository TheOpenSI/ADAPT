import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import pandas as pd

from dataset_builder.mmlu_builder_base import MMLUBenchmarkBuilderBase
from config import SERVICE_SETS


class MMLUValidationBuilder(MMLUBenchmarkBuilderBase):

    def __init__(
        self,
        split="validation",
        per_specialist_sample=10,
        per_general_subject_sample=2,
        random_state=42
    ):

        super().__init__(split)

        self.per_specialist_sample = per_specialist_sample
        self.per_general_subject_sample = per_general_subject_sample
        self.random_state = random_state


    def _sample_profiling_data(self, df):
        """
        Sample 10 validation examples from each specialist service
        and 2 examples from each general-source subject.
        """
        print("\n[4] Sampling profiling examples...")

        specialist_df = df[
            df["expected_service"] != "general"
        ]

        general_df = df[
            df["expected_service"] == "general"
        ]

        sampled_specialists = (
            specialist_df
            .groupby("subject", group_keys=False)
            .sample(
                n=self.per_specialist_sample,
                random_state=self.random_state
            )
        )

        sampled_general = (
            general_df
            .groupby("subject", group_keys=False)
            .sample(
                n=self.per_general_subject_sample,
                random_state=self.random_state
            )
        )

        profiling_data = pd.concat(
            [sampled_specialists, sampled_general],
            ignore_index=True
        )

        print("[4] Profiling subject counts:")
        print(profiling_data["subject"].value_counts())

        print("\n[4] Expected service counts:")
        print(profiling_data["expected_service"].value_counts())

        print(
            f"\n[4] Total profiling examples: "
            f"{len(profiling_data)}"
        )

        return profiling_data


    def _select_columns(self, df):
        """
        Keep columns required for profiling and
        service-description construction.
        """
        return df[
            [
                "question",
                "subject",
                "choices",
                "answer",
                "expected_service"
            ]
        ].copy()


    def _save_files(self, df):
        """
        Save the validation profiling dataset as CSV and JSON.
        """
        base_path = "../dataset/validation_profiling"

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
        print("BUILDING VALIDATION PROFILING DATASET")
        print("=" * 60)

        filtered_data = self._filter_subjects(subjects)

        converted_data = self._convert_subject(
            filtered_data,
            subjects
        )

        profiling_data = self._sample_profiling_data(
            converted_data
        )

        profiling_data = self._select_columns(
            profiling_data
        )

        return profiling_data


    def build_and_save(self, service_id, subjects):
        """
        Build and save the validation profiling dataset.
        """

        profiling_data = self._build_dataset(subjects)

        self._save_files(profiling_data)

        print(
            f"[DONE] Validation profiling dataset "
            f"for service set {service_id}"
        )

        print(
            f"[DONE] Total rows: {len(profiling_data)}"
        )

        return profiling_data


if __name__ == "__main__":

    builder = MMLUValidationBuilder()

    subjects = SERVICE_SETS[10]

    builder.build_and_save(
        service_id=10,
        subjects=subjects
    )