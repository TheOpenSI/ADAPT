import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")


from data.MMLU.specialists_dataset.mmlu_builder_base import MMLUBenchmarkBuilderBase
import pandas as pd
from dataset.config import SERVICE_SETS

class MMLUTestBuilder(MMLUBenchmarkBuilderBase):
    def __init__(
        self,
        split="test",
        per_specialist_subject_cap=100,
        # per_general_subject_cap=20,
        random_state=42
    ):
        super().__init__(split,include_general=False)

        self.per_specialist_subject_cap = per_specialist_subject_cap
        # self.per_general_subject_cap = per_general_subject_cap
        self.random_state = random_state


    def _cap_specialist(self,df):
        """
        Sample 100 questions from each specialist service.
        """
        print("\n[4] Sampling specialist services...")

        specialist_df = df[df["expected_service"] != "general"]
        sampled_specialist = (
            specialist_df.groupby("subject",group_keys = False).sample(
                n = self.per_specialist_subject_cap,
                random_state=self.random_state
                )
        ) 
        print("[4] Sampled specialist subject counts:")
        print(sampled_specialist["subject"].value_counts())

        return sampled_specialist


    # def _cap_general(self, df):
    #     """
    #     Sample a fixed number of questions from each general-source subject
    #     while keeping all specialist-service questions.
    #     """
    #     print("\n[5] Sampling general subjects...")

    #     general_df = df[df["expected_service"] == "general"]

    #     sampled_general = (
    #         general_df
    #         .groupby("subject", group_keys=False)
    #         .sample(
    #             n=self.per_general_subject_cap,
    #             random_state=self.random_state
    #         )
    #     )

    #     print("[5] Sampled general subject counts:")
    #     print(sampled_general["subject"].value_counts())

    #     return sampled_general


    # def _combine_data(self, df):

    #     sampled_specialist = self._cap_specialist(df)
    #     sampled_general = self._cap_general(df)

    #     print("\n[6] Combining specialist and general samples...")
    #     final_data = pd.concat(
    #         [sampled_specialist, sampled_general],
    #         ignore_index=True
    #     )

    #     print("[6] Expected service counts after sampling:")
    #     print(final_data["expected_service"].value_counts())

    #     return final_data


    def _select_columns(self, df):
        """
        Keep only the columns required for final routing evaluation.
        """
        return df[
            ["question", "subject", "expected_service"]
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

        print(f"\n[7] Saved: {base_path}.csv")
        print(f"[7] Saved: {base_path}.json")
    

    def _build_dataset(self, subjects):
        print("\n" + "=" * 60)
        print("BUILDING TEST DATASET")
        print("=" * 60)

        filtered_data = self._filter_subjects(subjects)

        converted_data = self._convert_subject(
            filtered_data,
            subjects
        )

        # sampled_data = self._combine_data(
        #     converted_data
        # )

        sampled_specialist = self._cap_specialist(converted_data)

        routing_data = self._select_columns(sampled_specialist)

        return routing_data

    def build_and_save(self, service_id, subjects):
        """
        Build and save the final test dataset for one service set.
        """

        routing_data = self._build_dataset(subjects)

        self._save_files(routing_data,service_id)

        print(f"\n[DONE] test_service_{service_id}")
        print(f"[DONE] Total rows: {len(routing_data)}")
        print(routing_data["expected_service"].value_counts())

        return routing_data


if __name__ == "__main__":

    builder = MMLUTestBuilder()

    for service_id, subjects in SERVICE_SETS.items():
        builder.build_and_save(
            service_id,
            subjects
        )