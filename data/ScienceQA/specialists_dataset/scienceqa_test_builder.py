"""
ScienceQATestBuilder

Responsibilities:
- build ScienceQA routing datasets from the test split
- sample 100 questions from each selected topic
- keep only the columns needed for routing experiments
- save benchmark datasets as JSON and CSV
- generate specialist routing benchmarks for K={3,5,8,10}
"""
import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../../..")


import pandas as pd
from data.ScienceQA.specialists_dataset.scienceqa_config import SCIENCEQA_SERVICE_SETS
from data.ScienceQA.specialists_dataset.scienceqa_builder_base import ScienceQABenchmarkBuilderBase

class ScienceQATestBuilder(ScienceQABenchmarkBuilderBase):
    def __init__(
            self,
            split,
            per_topic_cap,
            random_state
    ):
        super().__init__(split)
        self.per_topic_cap = per_topic_cap
        self.random_state = random_state


    def _cap_specialist(self, df):
        """
        Sample 100 questions from each specialist topic.
        """

        print("\n[4] Sampling specialist topics...")
        
        sampled_specialist = (
            df.groupby(
                "topic",
                group_keys=False
            ).sample(
                n=self.per_topic_cap,
                random_state=self.random_state)
            )
        
        print("[4] Sampled specialist topic counts:")
        print(sampled_specialist["topic"].value_counts())

        return sampled_specialist


    def _select_columns(self, df):
        """
        Keep only the columns required for routing evaluation.
        """

        return df[
            [
                "question",
                "subject",
                "expected_service"
            ]
        ].copy()

    def _save_files(self, df, service_id):
        """ 
        Save the final test dataset as CSV and JSON.
        """
        base_path = f"../specialists_dataset/test_service_{service_id}"

        df.to_csv(
            f"{base_path}.csv",
            index=False
        )

        df.to_json(
            f"{base_path}.json",
            orient="records",
            indent=2
        )

        print(f"\n[5] Saved: {base_path}.csv")
        print(f"[5] Saved: {base_path}.json")


    def _build_dataset(self, topics):

        print("\n" + "=" * 60)
        print("BUILDING SCIENCEQA TEST DATASET")
        print("=" * 60)

        filtered_data = self._filter_topics(topics)

        converted_data = self._convert_topic(filtered_data)

        sampled_specialist = self._cap_specialist(converted_data)
        
        routing_data = self._select_columns(sampled_specialist)

        return routing_data


    def build_and_save(
            self,
            service_id,
            topics
    ):
        routing_data = self._build_dataset(topics)
        
        self._save_files(routing_data, service_id)

        print(f"\n[DONE] test_service_{service_id}")
        print(f"[DONE] Total rows: {len(routing_data)}")

        return routing_data


if __name__ == "__main__":

    SPLIT = "test"
    PER_TOPIC_CAP = 100
    RANDOM_STATE = 42

    builder = ScienceQATestBuilder(
        split=SPLIT,
        per_topic_cap=PER_TOPIC_CAP,
        random_state=RANDOM_STATE
    )

    for service_id, topics in SCIENCEQA_SERVICE_SETS.items():
        builder.build_and_save(
            service_id,
            topics
        )






