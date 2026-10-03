"""
ScienceQABenchmarkBuilderBase

Responsibilities:
- load_dataset("derek-thomas/ScienceQA")
- keep only selected ScienceQA topics
- convert topic -> expected_service
- provide common utilities used by ScienceQA benchmark builders (ex: test set or validation set)
"""

import pandas as pd
from datasets import load_dataset
from abc import ABC, abstractmethod

class ScienceQABenchmarkBuilderBase(ABC):
    def __init__(self, split):
        self.split = split
        self.df = self._load_official_dataset()


    def _load_official_dataset(self):
        dataset = load_dataset(
            "derek-thomas/ScienceQA",
            split=self.split
        )
        
        df = pd.DataFrame(dataset)
        print(f"[1] Full dataset loaded: {df.shape}")
        print(f"[1] Number of topics: {df['topic'].nunique()}")

        print("[1] Topics:")
        print(sorted(df["topic"].unique()))

        return df

    def _filter_topics(self, topics):
        """
        Keep only selected specialist topics.
        """

        filtered = self.df[
        self.df["topic"].isin(topics)].copy()

        print("[2] Selected topics:")
        print(topics)
        
        print(f"[2] Filtered dataset shape: {filtered.shape}")
        
        print("[2] Topic counts:")
        print(filtered["topic"].value_counts())

        return filtered

    def _convert_topic(self, df):
        """
        Convert ScienceQA topic labels to routing labels.
        For ScienceQA, each topic is its own specialist.
        """

        df["expected_service"] = df["topic"]

        # Align with MMLU schema
        df["subject"] = df["topic"]

        print("[3] Expected service counts:")
        print(df["expected_service"].value_counts())

        return df

    @abstractmethod
    def _build_dataset(self, topics):
        pass

    @abstractmethod
    def build_and_save(self, service_id, topics):
        pass

