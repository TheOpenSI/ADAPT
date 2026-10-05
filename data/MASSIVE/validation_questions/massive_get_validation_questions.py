"""
GetMASSIVEValidationQuestions

Responsibilities:
- build MASSIVE validation datasets
- filter validation utterances to selected specialist subjects
- keep all available validation utterances for concept extraction
- preserve subject -> expected_service mappings
- save the validation dataset as JSON and CSV
"""

import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/../../.."
)

from data.MASSIVE.specialists.massive_builder_base import (
    MASSIVEBenchmarkBuilderBase
)

from data.MASSIVE.specialists.massive_config import (
    MASSIVE_SERVICE_SETS
)


class GetMASSIVEValidationQuestions(
    MASSIVEBenchmarkBuilderBase
):

    def __init__(self, split="validation"):
        super().__init__(split)

    def _select_columns(self, df):
        """
        Keep columns required for concept extraction
        and specialist profile generation.
        """

        return df[
            [
                "text",
                "subject",
                "expected_service"
            ]
        ].rename(
            columns={
                "text": "question"
            }
        )

    def _save_files(
        self,
        df,
        service_id
    ):
        """
        Save validation questions as CSV and JSON.
        """

        base_path = (
            "../validation_questions/"
            f"massive_validation_questions_{service_id}"
        )

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

    def _build_dataset(
        self,
        subjects
    ):
        print("\n" + "=" * 60)
        print("BUILDING MASSIVE VALIDATION DATASET")
        print("=" * 60)

        filtered_data = (
            self._filter_subjects(
                subjects
            )
        )

        converted_data = (
            self._convert_subject(
                filtered_data
            )
        )

        profiling_data = (
            self._select_columns(
                converted_data
            )
        )

        print(
            "\n[3] Validation examples per subject:"
        )

        print(
            profiling_data[
                "expected_service"
            ].value_counts()
        )

        return profiling_data

    def build_and_save(
        self,
        service_id,
        subjects
    ):
        """
        Build and save validation dataset.
        """

        profiling_data = (
            self._build_dataset(
                subjects
            )
        )

        self._save_files(
            profiling_data,
            service_id
        )

        print(
            f"\n[DONE] Validation dataset "
            f"for service set {service_id}"
        )

        print(
            f"[DONE] Total rows: "
            f"{len(profiling_data)}"
        )

        print(
            "\nExpected service counts:"
        )

        print(
            profiling_data[
                "expected_service"
            ].value_counts()
        )

        return profiling_data


if __name__ == "__main__":

    SERVICE_ID = 9

    builder = GetMASSIVEValidationQuestions(
        split="validation"
    )

    subjects = MASSIVE_SERVICE_SETS[
        SERVICE_ID
    ]

    builder.build_and_save(
        service_id=SERVICE_ID,
        subjects=subjects
    )