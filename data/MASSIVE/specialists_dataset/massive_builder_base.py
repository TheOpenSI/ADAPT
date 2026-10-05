"""
MASSIVEBenchmarkBuilderBase

Responsibilities:
- load the MTEB MASSIVE dataset
- keep only selected MASSIVE specialist subjects
- convert subject labels to routing labels
- provide common utilities used by MASSIVE benchmark builders
"""

from abc import ABC, abstractmethod

import pandas as pd
from datasets import load_dataset


class MASSIVEBenchmarkBuilderBase(ABC):

    def __init__(self, split):
        self.split = split
        self.df = self._load_official_dataset()

    def _load_official_dataset(self):

        dataset = load_dataset(
            "mteb/amazon_massive_intent",
            split=self.split
        )

        df = pd.DataFrame(dataset)

        # English only
        df = df[
            df["lang"] == "en"
        ].copy()

        print(f"[1] Full dataset loaded: {df.shape}")

        print(
            f"[1] Number of subjects: "
            f"{df['label'].nunique()}"
        )

        print("[1] Subjects:")

        print(
            sorted(
                df["label"]
                .dropna()
                .unique()
            )
        )

        return df

    def _filter_subjects(
        self,
        subjects
    ):
        """
        Keep only selected MASSIVE specialist subjects.
        """

        filtered = self.df[
            self.df["label"].isin(subjects)
        ].copy()

        print("[2] Selected subjects:")
        print(subjects)

        print(
            f"[2] Filtered dataset shape: "
            f"{filtered.shape}"
        )

        print("[2] Subject counts:")

        print(
            filtered["label"]
            .value_counts()
            .reindex(subjects, fill_value=0)
        )

        return filtered

    def _convert_subject(
        self,
        df
    ):
        """
        Convert MASSIVE subject labels
        to routing labels.
        """

        df = df.copy()

        df["expected_service"] = df["label"]

        # Align with MMLU and MedMCQA
        df["subject"] = df["label"]

        print("[3] Expected service counts:")

        print(
            df["expected_service"]
            .value_counts()
        )

        return df

    @abstractmethod
    def _build_dataset(
        self,
        subjects
    ):
        pass

    @abstractmethod
    def build_and_save(
        self,
        service_id,
        subjects
    ):
        pass