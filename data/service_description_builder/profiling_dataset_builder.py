import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from datasets import load_dataset
import pandas as pd

from dataset_builder.config import SERVICE_SETS, GENERAL_SUBJECTS
from pathlib import Path


class ProfilingDatasetBuilder:
    """
    Data preparation: Build a small, balanced profiling dataset from MMLU validation data.
    """


    def __init__(self, subjects, results_dir = "../data",specialist_cap=10, general_cap=2, random_state=42):
        self.subjects = subjects
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.specialist_cap = specialist_cap
        self.general_cap = general_cap
        self.random_state = random_state


    def _load_validation_dataset(self):
        dataset = load_dataset(
            "cais/mmlu",
            "all",
            split="validation"
        )
        return pd.DataFrame(dataset)
    

    def _filter_subjects(self, df):
        allowed_subjects = self.subjects + GENERAL_SUBJECTS

        filtered_df = df[df["subject"].isin(allowed_subjects)].copy()

        return filtered_df


    def _sample_questions(self, df):
        specialist_df = (
            df[df["subject"].isin(self.subjects)]
            .groupby("subject", group_keys=False)
            .sample(n=self.specialist_cap, random_state=self.random_state)
            )
        general_df = (
            df[df["subject"].isin(GENERAL_SUBJECTS)]
            .groupby("subject", group_keys=False)
            .sample(n=self.general_cap, random_state=self.random_state)
            )

        return pd.concat([specialist_df, general_df],ignore_index=True)

    def _create_expected_service(self, df):
        df["expected_service"] = df["subject"].where(
            df["subject"].isin(self.subjects),
            "general"
            )
        return df


    def _shuffle_dataset(self, df):
        return (df.sample(frac=1,random_state=42).reset_index(drop=True))


    def _select_columns(self, df):
        return df[["question","choices","subject","expected_service"]].copy()


    def _save_files(self, df):
        path = self.results_dir / f"profiling_dataset_services{len(self.subjects)}"

        df.to_csv(f"{path}.csv", index=False)
        df.to_json(f"{path}.json", orient="records", indent=2)

        return path


    def build_dataset(self):
        df = self._load_validation_dataset()
        print(f"[1] Loaded validation dataset: {len(df)} rows")

        df = self._filter_subjects(df)
        print(f"[2] Filtered to allowed subjects: {len(df)} rows")
        print(f"    Subjects present: {sorted(df['subject'].unique())}")

        df = self._sample_questions(df)
        print(f"[3] Sampled questions: {len(df)} rows "
              f"({self.specialist_cap}/specialist, {self.general_cap}/general subject)")
        
        df = self._create_expected_service(df)
        print(f"[4] Created expected_service")
        print(f"    {df['expected_service'].value_counts().to_dict()}")

        df = self._shuffle_dataset(df)
        print(f"[5] Shuffled dataset")

        df = self._select_columns(df)
        print(f"[6] Selected columns: {df.columns.tolist()}")

        save_path = self._save_files(df)
        print(f"[7] Profiling data saved to {save_path}.csv / {save_path}.json")

        return df

# if __name__ == "__main__":
#     builder = ProfilingDatasetBuilder(subjects=SERVICE_SETS[3])
#     df = builder.build_dataset()