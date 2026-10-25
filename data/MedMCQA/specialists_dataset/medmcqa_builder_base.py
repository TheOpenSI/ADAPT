"""
MedMCQABenchmarkBuilderBase

Responsibilities:
- load the official MedMCQA dataset
- keep only selected MedMCQA subjects
- convert subject_name to routing labels
- provide common utilities used by MedMCQA benchmark builders
"""

from abc import ABC, abstractmethod

import pandas as pd
from datasets import load_dataset


class MedMCQABenchmarkBuilderBase(ABC):

    def __init__(self, split):
        self.split = split
        self.df = self._load_official_dataset()

    def _load_official_dataset(self):
        dataset = load_dataset(
            "openlifescienceai/medmcqa",
            split=self.split
        )

        df = pd.DataFrame(dataset)

        print(f"[1] Full dataset loaded: {df.shape}")
        print(
            f"[1] Number of subjects: "
            f"{df['subject_name'].nunique()}"
        )

        print("[1] Subjects:")
        print(
            sorted(
                df["subject_name"]
                .dropna()
                .unique()
            )
        )

        return df

    def _filter_subjects(self, subjects):
        """
        Keep only selected MedMCQA specialist subjects.
        """

        filtered = self.df[
            self.df["subject_name"].isin(subjects)
        ].copy()

        print("[2] Selected subjects:")
        print(subjects)

        print(f"[2] Filtered dataset shape: {filtered.shape}")

        print("[2] Subject counts:")
        print(
            filtered["subject_name"]
            .value_counts()
            .reindex(subjects, fill_value=0)
        )

        return filtered

    def _convert_subject(self, df):
        """
        Convert MedMCQA subject labels to routing labels.

        Each selected subject is treated as one specialist.
        """

        df = df.copy()

        df["expected_service"] = df["subject_name"]

        # Align with the existing MMLU routing schema.
        df["subject"] = df["subject_name"]

        print("[3] Expected service counts:")
        print(df["expected_service"].value_counts())

        return df

    @abstractmethod
    def _build_dataset(self, subjects):
        pass

    @abstractmethod
    def build_and_save(self, service_id, subjects):
        pass