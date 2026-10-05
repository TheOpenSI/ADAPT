"""
MASSIVETestBuilder

Responsibilities:
- build MASSIVE routing datasets from the test split
- sample a fixed number of utterances from each selected service
- keep only the columns needed for routing experiments
- save benchmark datasets as JSON and CSV
- generate specialist routing benchmarks for K={3,5,7,9}
"""

import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/../../.."
)

import pandas as pd

from data.MASSIVE.specialists_dataset.massive_config import (
    MASSIVE_SERVICE_SETS
)
from data.MASSIVE.specialists_dataset.massive_builder_base import (
    MASSIVEBenchmarkBuilderBase
)


class MASSIVETestBuilder(MASSIVEBenchmarkBuilderBase):

    def __init__(
            self,
            split,
            per_subject_cap,
            random_state
    ):
        super().__init__(split)

        self.per_subject_cap = per_subject_cap
        self.random_state = random_state

    def _cap_specialists(self, df):
        """
        Sample the configured number of questions from each
        selected MASSIVE subject.
        """

        print("\n[4] Sampling specialist subjects...")

        subject_counts = df["label"].value_counts()

        insufficient_subjects = subject_counts[
            subject_counts < self.per_subject_cap
        ]

        if not insufficient_subjects.empty:
            raise ValueError(
                "The following subjects do not contain enough "
                f"'{self.split}' examples for a cap of "
                f"{self.per_subject_cap}:\n"
                f"{insufficient_subjects}"
            )

        sampled_specialists = (
            df.groupby(
                "label",
                group_keys=False
            )
            .sample(
                n=self.per_subject_cap,
                random_state=self.random_state
            )
            .reset_index(drop=True)
        )

        print("[4] Sampled specialist subject counts:")
        print(
            sampled_specialists["label"]
            .value_counts()
        )

        return sampled_specialists

    def _select_columns(self, df):
        """
        Keep only the columns required for routing evaluation.
        Align MASSIVE schema with MedMCQA/MMLU.
        """

        routing_df = df[
        [
            "text",
            "subject",
            "expected_service"
        ]
    ].copy()

        routing_df = routing_df.rename(
        columns={
            "text": "question"
        }
    )

        return routing_df

    def _save_files(self, df, service_id):
        """
        Save the final test dataset as CSV and JSON.
        """

        base_path = (
            f"../specialists_dataset/"
            f"test_service_{service_id}"
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

        print(f"\n[5] Saved: {base_path}.csv")
        print(f"[5] Saved: {base_path}.json")

    def _build_dataset(self, subjects):

        print("\n" + "=" * 60)
        print("BUILDING MEDMCQA TEST DATASET")
        print(f"SPLIT: {self.split}")
        print("=" * 60)

        filtered_data = self._filter_subjects(
            subjects
        )

        converted_data = self._convert_subject(
            filtered_data
        )

        sampled_specialists = self._cap_specialists(
            converted_data
        )

        routing_data = self._select_columns(
            sampled_specialists
        )

        return routing_data

    def build_and_save(
            self,
            service_id,
            subjects
    ):
        routing_data = self._build_dataset(
            subjects
        )

        self._save_files(
            routing_data,
            service_id
        )

        print(
            f"\n[DONE] test_service_{service_id}"
        )
        print(
            f"[DONE] Total rows: {len(routing_data)}"
        )

        return routing_data



if __name__ == "__main__":

    SPLIT = "test"
    PER_SUBJECT_CAP = 22
    RANDOM_STATE = 42

    builder = MASSIVETestBuilder(
        split=SPLIT,
        per_subject_cap=PER_SUBJECT_CAP,
        random_state=RANDOM_STATE
    )

    # Inspect the number of test questions available
    # for every MedMCQA subject.
    # test_counts = (
    #     builder.df["label"]
    #     .value_counts()
    #     .sort_values(ascending=False)
    # )

    # print("\nAll test subject counts:")
    # print(test_counts)

    # print(
    #     f"\nSubjects eligible for "
    #     f"{PER_SUBJECT_CAP} test questions:"
    # )

    # print(
    #     test_counts[
    #         test_counts >= PER_SUBJECT_CAP
    #     ]
    # )

    # Temporarily disable dataset generation.
    # The current configuration contains subjects
    # with fewer than 100 test questions.
    #
    for service_id, subjects in (
        MASSIVE_SERVICE_SETS.items()
    ):
        builder.build_and_save(
            service_id,
            subjects
        )