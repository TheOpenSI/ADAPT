import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")


import pandas as pd
from dataset.config import SERVICE_SETS

class EvaluationBuilder:

    def __init__(self, split):
        self.split = split

    def _load_specialists(self, service_id):

        file_path = (
            f"../specialists_dataset/"
            f"{self.split}_service_{service_id}.csv"
        )

        return pd.read_csv(file_path)

    def _load_general(self):

        file_path = (
            f"../general_dataset/"
            f"{self.split}_external_general.csv"
        )

        return pd.read_csv(file_path)

    def _combine(self, service_id):

        specialists = self._load_specialists(
            service_id
        )

        general = self._load_general()

        evaluation = pd.concat(
            [specialists, general],
            ignore_index=True
        )

        return evaluation

    def _save(self, df, service_id):

        file_path = (
            f"../evaluation_dataset/"
            f"{self.split}_eval_service_{service_id}.csv"
        )

        df.to_csv(
            file_path,
            index=False
        )

        print(f"Saved: {file_path}")

    def build_and_save(self, service_id):

        evaluation = self._combine(
            service_id
        )

        self._save(
            evaluation,
            service_id
        )

        print(
            evaluation["expected_service"]
            .value_counts()
        )

        return evaluation


if __name__ == "__main__":

    builder = EvaluationBuilder(
        split="dev"
    )

    for service_id in SERVICE_SETS.keys():

        builder.build_and_save(
            service_id
        )