import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from dataset.config import SERVICE_SETS
from data.MMLU.specialists_dataset.mmlu_builder_base import MMLUBenchmarkBuilderBase


class MMLUDevBuilder(MMLUBenchmarkBuilderBase):

    def __init__(self, split="dev"):
        super().__init__(split=split, include_general=False)


    def _select_columns(self, df):
        """
        Keep columns needed for threshold calibration.
        """
        return df[["question", "subject", "expected_service"]].copy()

    def _save_files(self, df, service_id):
        """
        Save the dev calibration dataset as CSV and JSON.
        """

        base_path = f"../specialists_dataset/dev_service_{service_id}"
    
        df.to_csv(f"{base_path}.csv",index=False)
    
        # df.to_json(f"{base_path}.json", orient="records", indent=2)
    
        print(f"[4] Saved: {base_path}.csv")
        print(f"[4] Saved: {base_path}.json")


    def _build_dataset(self, subjects):
        print("\n" + "=" * 60)
        print("BUILDING DEV PROFILING DATASET")
        print("=" * 60)

        # Includes specialists + General-source subjects
        filtered_data = self._filter_subjects(subjects)

        # Specialists keep their names
        # General-source subjects -> "general"
        converted_data = self._convert_subject(filtered_data, subjects)
    
        dev_data = self._select_columns(converted_data)
        return dev_data


    def build_and_save(self, service_id, subjects):

        dev_data = self._build_dataset(subjects)

        self._save_files(dev_data, service_id)

        print(f"\n[DONE] dev_service_{service_id}")
        print(f"[DONE] Total rows: {len(dev_data)}")


        print("\nExpected service counts:")
        print(dev_data["expected_service"].value_counts())

        print("\nOriginal subject counts:")
        print(dev_data["subject"].value_counts())

        return dev_data


if __name__ == "__main__":
    builder = MMLUDevBuilder()

    for service_id, subjects in SERVICE_SETS.items():
        builder.build_and_save(service_id=service_id, subjects=subjects)
        

