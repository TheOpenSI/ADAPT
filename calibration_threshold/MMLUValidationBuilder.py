import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)

from data.dataset_builder.mmlu_builder_base import (
    MMLUBenchmarkBuilderBase
)

from data.dataset_builder.config import SERVICE_SETS


class MMLUValidationBenchmarkBuilder(
    MMLUBenchmarkBuilderBase
):

    def __init__(self):
        super().__init__(
            split="validation",
            include_general=True
        )

    def _select_columns(self, df):
        """
        Keep only columns required for routing calibration.
        """

        return df[
            [
                "question",
                "expected_service"
            ]
        ].copy()

    def _save_files(self, df, service_id):
        """
        Save calibration datasets.
        """

        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir,"dataset")
        os.makedirs(output_dir, exist_ok=True)

        base_path = os.path.join(output_dir, f"validation_service_{service_id}")

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
            f"\n[4] Saved: {base_path}.csv"
        )

        print(
            f"[4] Saved: {base_path}.json"
        )

    def _build_dataset(self, subjects):

        print("\n" + "=" * 60)
        print("BUILDING VALIDATION DATASET")
        print("=" * 60)

        filtered_data = self._filter_subjects(
            subjects
        )

        converted_data = self._convert_subject(
            filtered_data,
            subjects
        )

        routing_data = self._select_columns(
            converted_data
        )

        return routing_data

    def build_and_save(
        self,
        service_id,
        subjects
    ):
        """
        Build and save validation routing dataset.
        """

        routing_data = self._build_dataset(
            subjects
        )

        self._save_files(
            routing_data,
            service_id
        )

        print(
            f"\n[DONE] validation_service_{service_id}"
        )

        print(
            f"[DONE] Total rows: {len(routing_data)}"
        )

        print(
            routing_data[
                "expected_service"
            ].value_counts()
        )

        return routing_data


if __name__ == "__main__":

    builder = MMLUValidationBenchmarkBuilder()

    for service_id, subjects in (
        SERVICE_SETS.items()
    ):
        builder.build_and_save(
            service_id,
            subjects
        )