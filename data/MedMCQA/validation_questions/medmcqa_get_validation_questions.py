"""
GetMedMCQAValidationQuestions

Responsibilities:
- build MedMCQA validation datasets from the validation split
- filter validation questions to the selected specialist subjects
- keep all available validation questions
- keep columns required for concept extraction and profile generation
- preserve subject_name -> expected_service mappings
- save the validation dataset as JSON and CSV
"""

import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/../../.."
)

from data.MedMCQA.specialists_dataset.medmcqa_builder_base import (
    MedMCQABenchmarkBuilderBase
)
from data.MedMCQA.specialists_dataset.medmcqa_config import (
    MEDMCQA_SERVICE_SETS
)


class GetMedMCQAValidationQuestions(MedMCQABenchmarkBuilderBase):

    def __init__(
        self,
        split="validation"
    ):
        super().__init__(split)

    def _select_columns(self, df):
        """
        Keep columns required for profiling and
        service-description construction.
        """

        df = df.copy()

        # Convert MedMCQA's separate options into
        # the same choices structure used by MMLU.
        df["choices"] = df[
            [
                "opa",
                "opb",
                "opc",
                "opd"
            ]
        ].values.tolist()

        return df[
            [
                "question",
                "subject",
                "choices",
                "expected_service"
            ]
        ].copy()


    def _cap_subjects(
        self,
        df,
        max_questions=50,
    ):
        """
        Cap each subject to a maximum number of questions.
        """
        return (
            df.groupby(
                "expected_service",
                group_keys=False
            )
            .apply(
                lambda x: x.sample(
                    n=min(len(x), max_questions),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )

    def _save_files(
        self,
        df,
        service_id
    ):
        """
        Save the validation profiling dataset as CSV and JSON.
        """

        base_path = (
            "../validation_questions/"
            f"medmcqa_validation_questions_{service_id}"
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
        print("BUILDING MEDMCQA VALIDATION DATASET")
        print("=" * 60)

        # Specialist subjects only.
        filtered_data = self._filter_subjects(
            subjects
        )

        # expected_service remains the subject name.
        converted_data = self._convert_subject(
            filtered_data
        )

        profiling_data = self._select_columns(
            converted_data
        )

        profiling_data = self._cap_subjects(
            profiling_data,
            max_questions=50
        )

        return profiling_data

    
    def build_and_save(
        self,
        service_id,
        subjects
    ):
        """
        Build and save the validation profiling dataset.
        """

        profiling_data = self._build_dataset(
            subjects
        )

        self._save_files(
            profiling_data,
            service_id
        )

        print(
            f"\n[DONE] Validation profiling dataset "
            f"for service set {service_id}"
        )

        print(
            f"[DONE] Total rows: "
            f"{len(profiling_data)}"
        )

        print("\nExpected service counts:")
        print(
            profiling_data[
                "expected_service"
            ].value_counts()
        )

        return profiling_data


if __name__ == "__main__":

    SERVICE_ID = 9

    builder = GetMedMCQAValidationQuestions(
        split="validation"
    )

    subjects = MEDMCQA_SERVICE_SETS[
        SERVICE_ID
    ]

    builder.build_and_save(
        service_id=SERVICE_ID,
        subjects=subjects
    )