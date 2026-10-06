import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")


import pandas as pd
from datasets import load_dataset


class ExternalGeneralBuilder:
    def __init__(self, 
                 banking_samples=25,
                 clinc_samples=25,
                 banking_split="train",
                 clinc_split="train",
                 seed=42):
        
        self.banking_samples = banking_samples
        self.clinc_samples = clinc_samples

        self.banking_split = banking_split
        self.clinc_split = clinc_split

        self.seed = seed

    def _get_banking_data(self):

        banking_dataset = load_dataset(
            "gtfintechlab/banking77",
            split= self.banking_split
        )

        banking_df = pd.DataFrame(banking_dataset)

        banking_df = banking_df.sample(
            n=self.banking_samples,
            random_state=self.seed
        ).reset_index(drop=True)

        banking = pd.DataFrame()

        banking["question"] = banking_df["text"]
        banking["subject"] = "banking77"
        banking["expected_service"] = "general"

        return banking


    def _get_clinc_data(self):

        clinc_dataset = load_dataset(
            "clinc/clinc_oos",
            "plus",
            split=self.clinc_split
        )

        clinc_df = pd.DataFrame(clinc_dataset)

        clinc_df = clinc_df.sample(
            n=self.clinc_samples,
            random_state=self.seed
        ).reset_index(drop=True)

        clinc = pd.DataFrame()

        clinc["question"] = clinc_df["text"]
        clinc["subject"] = "clinc_oos"
        clinc["expected_service"] = "general"

        return clinc


    def _combine_data(self):

        banking = self._get_banking_data()
        clinc = self._get_clinc_data()

        external_general = pd.concat([banking,clinc], ignore_index= True)

        # shuffle the order
        external_general = external_general.sample(
            frac=1,
            random_state=self.seed
        ).reset_index(drop=True)

        return external_general


    def _save_file(self, df, dataset_name):

        file_path = f"../general_dataset/{dataset_name}.csv"

        df.to_csv(
            file_path,
            index=False
        )

        print(f"\nSaved: {file_path}")

    def build_and_save(self, dataset_name):

        external_general = self._combine_data()

        self._save_file(
            external_general,
            dataset_name
        )

        print(
            f"\n[DONE] Total rows: "
            f"{len(external_general)}"
        )

        return external_general


if __name__ == "__main__":
    builder = ExternalGeneralBuilder(
        banking_samples=50,
        clinc_samples=50,
        banking_split="train",
        clinc_split="train",
        seed=42
)
    

    builder.build_and_save(
        dataset_name="mmlu_validation_external_general"
)















