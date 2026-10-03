"""
GetScienceQAValidationQuestions

Responsibilities:
- build ScienceQA validation datasets from the validation split
- filter validation questions to the selected specialist topics
- keep columns required for concept extraction and profile generation
- preserve topic -> expected_service mappings
- save validation datasets as JSON and CSV
"""

import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")

from data.ScienceQA.specialists_dataset.scienceqa_builder_base import ScienceQABenchmarkBuilderBase
from data.ScienceQA.specialists_dataset.config import SCIENCEQA_SERVICE_SETS

class GetScienceQAValidationQuestions(ScienceQABenchmarkBuilderBase):
    
    def __init__(self, split="validation"):
        super().__init__(split)


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

    
    def _cap_specialist(self, df):
            """
            Sample 100 validation questions from each specialist subject.
            """
    
            print("\n[3] Sampling validation subjects...")
            
            sampled_specialist = (
                df.groupby(
                    "topic",
                    group_keys=False
                ).sample(
                    n=100,
                    random_state=42)
                )
            
            print("[3] Sampled validation subject counts:")
            print(sampled_specialist["subject"].value_counts())
    
            return sampled_specialist


    def _save_files(self, df):
        """
        Save the validation profiling dataset as CSV and JSON.
        """
        base_path = "../validation_questions/scienceqa_validation_questions"

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


    def _build_dataset(self, topics):
        print("\n" + "=" * 60)
        print("BUILDING SCIENCEQA VALIDATION DATASET")
        print("=" * 60)

       # Specialist topics only
        filtered_data = self._filter_topics(topics)

        # expected_service will remain the specialist topic name
        converted_data = self._convert_topic(
            filtered_data
        )

        sampled_specialist = self._cap_specialist(
             converted_data
        )

        profiling_data = self._select_columns(
            sampled_specialist
        )

        return profiling_data


    def build_and_save(self, service_id, topics):

        """
        Build and save the validation profiling dataset.
        """

        profiling_data = self._build_dataset(topics)

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

    builder = GetScienceQAValidationQuestions()

    topics = SCIENCEQA_SERVICE_SETS[10]

    builder.build_and_save(
        service_id=10,
        topics=topics
    )        