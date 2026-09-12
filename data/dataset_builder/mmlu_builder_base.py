import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")


import pandas as pd
from datasets import load_dataset
from abc import ABC, abstractmethod
from config import GENERAL_SUBJECTS


class MMLUBenchmarkBuilderBase (ABC):
    def __init__(self, split, include_general=False):
        self.split = split
        self.include_general = include_general

        self.df = self._load_official_dataset()


    def _load_official_dataset(self):
        """Get all mmlu test dataset"""
        dataset = load_dataset("cais/mmlu","all",split=self.split)
        df = pd.DataFrame(dataset)

        print(f"[1] Full dataset loaded: {df.shape}")
        print(f"[1] Number of subjects: {df['subject'].nunique()}")
        return df


    def _filter_subjects(self, subjects):
        """
        Keep selected specialist subjects plus general-source subjects.
        """
        print("\n[2] Filtering subjects...")

        if self.include_general:
            allowed_subjects = subjects + GENERAL_SUBJECTS
        else:
            allowed_subjects = subjects

        print("[2] Specialist subjects:")
        print(subjects)

        if self.include_general:
            print("[2] General-source subjects:")
            print(GENERAL_SUBJECTS)

        filtered_data = self.df[self.df["subject"].isin(allowed_subjects)].copy()
        print(f"[2] Filtered dataset shape: {filtered_data.shape}")

        print("[2] Original subject counts:")
        print(filtered_data["subject"].value_counts())
  
        return filtered_data


    def _convert_subject(self,df,subjects):
        """
        Specialist subjects keep their names.
        General-source subjects become 'general'.
        """
        print("\n[3] Converting subjects to routing labels...")

        df["expected_service"] = df["subject"].where(
            df["subject"].isin(subjects),
            "general"
        )

        print("[3] Expected service counts:")
        print(df["expected_service"].value_counts())

        if self.include_general:
            print("\n[3] Subjects inside 'general':")
            print(df[df["expected_service"] == "general"]["subject"].value_counts())
        
        return df

    @abstractmethod
    def _build_dataset(self,subjects):
        pass

    @abstractmethod
    def build_and_save(self,service_id, subjects):
        pass
