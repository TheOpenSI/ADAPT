import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import pandas as pd
from datasets import load_dataset


class ExternalGeneralBuilder:
    def __init__(self,
                 banking_samples=100,
                 clinc_samples=100,
                 banking_split="test",
                 clinc_split="test",
                 seed=42,
                 out_dir="../general_dataset"):

        self.banking_samples = banking_samples
        self.clinc_samples = clinc_samples

        self.banking_split = banking_split
        self.clinc_split = clinc_split

        self.seed = seed
        self.out_dir = out_dir

    def _get_banking_data(self):

        banking_dataset = load_dataset(
            "gtfintechlab/banking77",
            split=self.banking_split
        )

        banking_df = pd.DataFrame(banking_dataset)

        n = min(self.banking_samples, len(banking_df))
        banking_df = banking_df.sample(
            n=n,
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

        n = min(self.clinc_samples, len(clinc_df))
        clinc_df = clinc_df.sample(
            n=n,
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

        external_general = pd.concat([clinc, banking], ignore_index=True)

        # remove repeated texts
        external_general = external_general.drop_duplicates("question")

        # shuffle the order
        external_general = external_general.sample(
            frac=1,
            random_state=self.seed
        ).reset_index(drop=True)

        return external_general

    def _save_file(self, df, dataset_name):

        os.makedirs(self.out_dir, exist_ok=True)
        file_path = f"{self.out_dir}/{dataset_name}.csv"

        df.to_csv(
            file_path,
            index=False
        )

        print(f"Saved: {file_path}  ({len(df)} rows)")

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

    # ------------------------------------------------------------------
    # Proportional design: K specialists get g * K general queries.
    # Every K uses different queries (disjoint blocks of one shuffled pool).
    # ------------------------------------------------------------------
    def build_proportional(self, ks, g=50, spare=200):
        """
        ks     list of pool sizes, e.g. [3, 5, 8, 10, 12, 14]
        g      general queries per specialist (general count = g * K)
        spare  extra queries drawn per source, to survive duplicate removal
        """

        need = sum(g * k for k in ks)
        self.banking_samples = need // 2 + spare
        self.clinc_samples = need // 2 + spare

        pool = self._combine_data()
        assert len(pool) >= need, f"only {len(pool)} general queries, need {need}"

        blocks = {}
        start = 0
        for k in ks:
            block = pool.iloc[start:start + g * k].reset_index(drop=True)
            start += g * k
            blocks[k] = block
            self._save_file(block, f"general_only_{k}")

        # no query may be used in two different K sets
        sets = {k: set(b["question"]) for k, b in blocks.items()}
        ks_list = list(ks)
        for i, a in enumerate(ks_list):
            for b in ks_list[i + 1:]:
                assert not (sets[a] & sets[b]), f"overlap between K={a} and K={b}"
        print("\n[CHECK] no query is shared between K sets:",
              {k: len(s) for k, s in sets.items()})

        return blocks


if __name__ == "__main__":
    builder = ExternalGeneralBuilder(
        banking_split="test",
        clinc_split="test",
        seed=42,
    )

    builder.build_proportional(
        ks=[3, 5, 8, 10, 12, 14],
        g=50,
    )