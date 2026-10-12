import os
import numpy as np
import pandas as pd


class TauNotebook:
    def __init__(self, dummy_general_csv, output_csv):
        # 1. take in the off-topic questions (dummy_general.csv)
        self.questions = pd.read_csv(dummy_general_csv)["question"].astype(str).tolist()
        self.csv_path = output_csv
        self.pages = {}                                        # specialist name -> its scores
        print(f"[notebook] read {len(self.questions)} off-topic questions from {os.path.basename(dummy_general_csv)}")

        # 2. load existing table
        if os.path.exists(output_csv):
            table = pd.read_csv(output_csv)

            # were the saved scores made with the same 100 questions, in the same order?
            if table["question"].astype(str).tolist() == self.questions:
                for name in table.columns[1:]:                  # every column except "question" = one specialist
                    self.pages[name] = table[name].tolist()     # keep its 100 scores in memory
                print(f"[notebook] found saved notebook, loaded {len(self.pages)} specialists: {list(self.pages)}")
            else:
                print("[notebook] The questions changed, so the old notebook is not used.")
        else:
            print(f"[notebook] no saved notebook yet at {output_csv}, starting empty")

    def tau(self, specialists, scorer):
        before = set(self.pages)    # names of specialists already in the notebook

        # 3. score only the specialists that are not in the notebook yet
        for name in specialists:
            if name not in self.pages:
                print(f"[notebook] {name}: not saved, asking the model to score {len(self.questions)} questions...")
                self.pages[name] = scorer(self.questions, name)
            else:
                print(f"[notebook] {name}: already saved, skipped")

        # 4. save: first column = question, then one column per specialist
        table = pd.DataFrame(self.pages)
        table.insert(0, "question", self.questions)
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        table.to_csv(self.csv_path, index=False)
        print(f"[notebook] saved {table.shape[0]} rows x {table.shape[1] - 1} specialists to {self.csv_path}")

        # 5. tau = mean of the MADs (MAD = median of |score - median|)
        mads = []
        for name in specialists:
            scores = np.array(self.pages[name])
            median = np.median(scores)
            abs_diff = np.abs(scores - median)
            mad = float(np.median(abs_diff))
            mads.append(mad)

            n = len(scores)
            print(f"\n--- {name} ---")
            print(f"median of the {n} scores = {median:.4f}")
            print(f"   {'#':>3} {'score':>10} {'|score - median|':>18}")
            for i, (sc, d) in enumerate(zip(scores, abs_diff), start=1):
                print(f"   {i:>3} {sc:>10.4f} {d:>18.4f}")
            print(f"MAD = median of the {n} |score - median| values = {mad:.4f}")

        tau = float(np.mean(mads))
        print(f"\nK={len(specialists)}  MAD of each specialist: " + ", ".join(f"{nm}={m:.4f}" for nm, m in zip(specialists, mads)))
        print(f"tau = ({' + '.join(f'{m:.4f}' for m in mads)}) / {len(mads)} = {tau:.4f}")
        print(f"K={len(specialists)}  tau={tau:.4f}  (scores asked for: {[n for n in specialists if n not in before]})")
        return tau

    def cutoff(self, specialists, tau, percentile=90, show=True):
        X = np.column_stack([self.pages[n] for n in specialists])   # rows = questions, columns = specialists
        top = X.max(axis=1)                                         # best score for each question
        average = X.mean(axis=1)                                    # mean of all scores for each question
        trust = (top - average) / tau
        value = float(np.percentile(trust, percentile))

        if show:
            above = int((trust > value).sum())
            print(f"\ntrust = (max score - mean score) / tau, for each of the {len(trust)} off-topic questions (tau = {tau:.4f})")
            print(f"   {'#':>3} {'max score':>10} {'mean score':>11} {'trust':>8}")
            for i in range(len(trust)):
                print(f"   {i+1:>3} {top[i]:>10.4f} {average[i]:>11.4f} {trust[i]:>8.4f}")
            print(f"\ntrust values as a list ({len(trust)} values):")
            print(trust.round(4).tolist())
            print(f"\n{percentile}th percentile of the trust values = {value:.4f} | questions above it: {above} of {len(trust)}")

        return value