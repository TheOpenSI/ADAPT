import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

from data.MMLU.specialists_dataset.mmlu_builder_base import MMLUBenchmarkBuilderBase
from data.dataset.config import SERVICE_SETS

class MMLUValidationBuilder(
    MMLUBenchmarkBuilderBase
):

    def __init__(self, split="validation"):

        super().__init__(
            split=split,
            include_general=False
        )

    def _select_columns(self, df):

        return df[
            [
                "question",
                "subject",
                "expected_service"
            ]
        ].copy()

    def _save_files(
        self,
        df,
        service_id
    ):

        base_path = (
            f"../specialists_dataset/"
            f"validation_service_{service_id}"
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

        print(
            f"[4] Saved: {base_path}.csv"
        )

        print(
            f"[4] Saved: {base_path}.json"
        )

    def _build_dataset(
        self,
        subjects
    ):

        print("\n" + "=" * 60)
        print(
            "BUILDING VALIDATION SPECIALIST DATASET"
        )
        print("=" * 60)

        filtered_data = (
            self._filter_subjects(
                subjects
            )
        )

        converted_data = (
            self._convert_subject(
                filtered_data,
                subjects
            )
        )

        validation_data = (
            self._select_columns(
                converted_data
            )
        )

        return validation_data

    def build_and_save(
        self,
        service_id,
        subjects
    ):

        validation_data = (
            self._build_dataset(
                subjects
            )
        )

        self._save_files(
            validation_data,
            service_id
        )

        print(
            f"\n[DONE] validation_service_{service_id}"
        )

        print(
            f"[DONE] Total rows: "
            f"{len(validation_data)}"
        )

        print(
            "\nExpected service counts:"
        )

        print(
            validation_data[
                "expected_service"
            ].value_counts()
        )

        print(
            "\nOriginal subject counts:"
        )

        print(
            validation_data[
                "subject"
            ].value_counts()
        )

        return validation_data


if __name__ == "__main__":

    builder = MMLUValidationBuilder()

    for service_id, subjects in SERVICE_SETS.items():

        builder.build_and_save(
            service_id=service_id,
            subjects=subjects
        )