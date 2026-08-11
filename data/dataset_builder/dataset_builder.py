from datasets import load_dataset
import pandas as pd

from config import SERVICE_SETS, GENERAL_SUBJECTS


class MMLUBenchmarkBuilder:
    def __init__(self, subjects, split="test", per_general_subject_cap=20, random_state=42):
        self.subjects = subjects
        self.split = split
        self.per_general_subject_cap = per_general_subject_cap
        self.random_state = random_state

    def _load_official_dataset(self):
        dataset = load_dataset("cais/mmlu","all",split=self.split)
        df = pd.DataFrame(dataset)
        return df

    def _filter_subjects(self, df):
        allowed_subjects = self.subjects + GENERAL_SUBJECTS
        filter_data = df[df["subject"].isin(allowed_subjects)].copy()
        return filter_data

    def _cap_general(self, df):
        is_general = df["expected_service"] == "general"

        capped_general = (
            df[is_general]
            .groupby("subject", group_keys=False)
            .sample(n=self.per_general_subject_cap, random_state=self.random_state)
        )

        return pd.concat([df[~is_general], capped_general], ignore_index=True)

    def _convert_subject(self,df):
        df["expected_service"] = df["subject"].where(
            df["subject"].isin(self.subjects),
            "general")
        return df

    def _select_columns(self, df):
        return df[["question", "expected_service"]].copy()

    def _save_files(self, df, path):
        df.to_csv(f"{path}.csv", index=False)
        df.to_json(f"{path}.json", orient="records", indent=2)


    def build_dataset(self):
        df = self._load_official_dataset()
        df = self._filter_subjects(df)
        df = self._convert_subject(df)
        df = self._cap_general(df)
        df = self._select_columns(df)
        self.dataset = df
        return df

if __name__ == "__main__":
    for service_id, subjects in SERVICE_SETS.items():
        builder = MMLUBenchmarkBuilder(subjects=subjects)
        df = builder.build_dataset()
        builder._save_files(df, path=f"../dataset/service_{service_id}")

        print(f"Service {service_id}: {len(df)} rows")
        print(df["expected_service"].value_counts())
        print()