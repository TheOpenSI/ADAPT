import ast
import os
import math
import numpy as np
import pandas as pd
from itertools import combinations


class MADGate:
    def __init__(self,
                 calibration_result_csv,                     # router output (gate off) on the 100 general reference queries
                 percentile=95,
                 save_dir=None
    ):
        self.percentile = percentile

        self.table = self.load_table(calibration_result_csv) # step 1: organise the table
        self.tau = self.find_tau()                     # step 2: the ruler
        self.curve = self.find_cutoffs()               # step 3: the cutoff for each K

        self.c0, self.c1 = self.fit_curve()            # step 4: cutoff(K) = c0 + c1 * ln K

        if save_dir:
            self.save(save_dir)

    # ---------------------------------------------------------------- step 1
    def load_table(self, train_general_csv):
        """General rows only, one column per specialist."""

        data = pd.read_csv(train_general_csv, usecols=["scores", "expected_service"])
        data = data[data["expected_service"] == "general"]

        # unpack scores
        table = data["scores"].apply(ast.literal_eval).apply(pd.Series)

        print(f"general rows = {len(table)} | specialists = {table.shape[1]}")
        if len(table) < 100:
            print(f"WARNING: only {len(table)} general rows; 100 or more keeps tau within about +-12%.")
        return table

    # ---------------------------------------------------------------- step 2
    def find_tau(self):
        """tau = mean of the MAD of each specialist column."""

        # MAD of each specialist column
        mads = []
        for column in self.table.columns:
            scores = self.table[column]
            middle = scores.median()                   # the column's typical (off-topic) score
            distances = (scores - middle).abs()        # how far each score is from the middle
            mads.append(distances.median())            # MAD = the typical distance

        # one ruler: the average of the column MADs
        tau = sum(mads) / len(mads)

        # the table, to check every number
        self.summary = pd.DataFrame({"middle (Median)": self.table.median().values, "MAD": mads},
                                    index=self.table.columns)

        print(self.summary)
        print(f"sum of MADs = {sum(mads):.4f} | tau = {tau:.4f}")

        return tau

    # ---------------------------------------------------------------- step 3
    def find_cutoffs(self, show=True):
        """For each K: the percentile of the general queries' trust, averaged over every group of K specialists."""

        rows = []
        for k in range(3, self.table.shape[1] + 1):
            cutoffs = []
            for columns in combinations(self.table.columns, k):
                group = self.table[list(columns)]                      # same rows, only k columns
                trust = (group.max(axis=1) - group.mean(axis=1)) / self.tau
                cutoffs.append(np.percentile(trust, self.percentile))

                if show and len(cutoffs) == 1:                         # show the first group of this K
                    print(f"\nK={k} | first group = {list(columns)}")
                    print(f"trust of the {len(trust)} general queries, sorted:")
                    print(np.sort(trust.values).round(2).tolist())
                    print(f"{self.percentile}th percentile of these = {cutoffs[0]:.2f}")

            rows.append((k, np.mean(cutoffs)))
            if show:
                print(f"K={k} | groups = {len(cutoffs)} | cutoff = average of the groups = {np.mean(cutoffs):.2f}")

        curve = pd.DataFrame(rows, columns=["K", "cutoff"])
        print()
        print(curve.round(4))
        return curve

    # ---------------------------------------------------------------- step 4
    def fit_curve(self):
        """A line through the points: cutoff = c0 + c1 * ln(K)."""

        c1, c0 = np.polyfit(np.log(self.curve["K"]), self.curve["cutoff"], 1)
        print(f"c0 = {c0:.4f} | c1 = {c1:.4f}")
        return float(c0), float(c1)

    def cutoff(self, k):
        """cutoff grows with the log of the number of specialists"""
        return self.c0 + self.c1 * math.log(k)

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        summary = self.summary.copy()
        summary.loc["tau"] = [None, self.tau]
        summary.to_csv(os.path.join(save_dir, "tau_table.csv"))
        self.curve.to_csv(os.path.join(save_dir, "cutoff_curve.csv"), index=False)
        print(f"saved to {save_dir}")


if __name__ == "__main__":
    dataset = "massive"
    model   = "Qwen/Qwen3-Reranker-4B"

    gate = MADGate(
        "../../calibration_threshold/results_off_topic_cal/massive/dummy_general/cross-encoder_Qwen-Qwen3-Reranker-4B.csv",
        percentile=90,
        save_dir=os.path.join("calibration_results", dataset, model.replace("/", "_")),
    )