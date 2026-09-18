import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import pandas as pd

from dataset_builder.mmlu_builder_base import MMLUBenchmarkBuilderBase
from data.dataset.config import SERVICE_SETS


class MMLUValidationBuilder(MMLUBenchmarkBuilderBase):

    def __init__(self, split="validation"):
        super().__init__(split, include_general=False)


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
                "expected_service"
            ]
        ].copy()


    def _save_files(self, df):
        """
        Save the validation profiling dataset as CSV and JSON.
        """
        base_path = "../dataset/validation_service_all"

        df.to_csv(
            f"{base_path}.csv",
            index=False
        )

        df.to_json(
            f"{base_path}.json",
            orient="records",
            indent=2
        )

        print(f"[4] Saved: {base_path}.csv")
        print(f"[4] Saved: {base_path}.json")


    def _build_dataset(self, subjects):
        print("\n" + "=" * 60)
        print("BUILDING SPECIALIST VALIDATION DATASET")
        print("=" * 60)

        # Specialist subjects only
        filtered_data = self._filter_subjects(subjects)

        # expected_service will remain the specialist subject name
        converted_data = self._convert_subject(
            filtered_data,
            subjects
        )

        profiling_data = self._select_columns(
            converted_data
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