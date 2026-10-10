import ast
import pandas as pd
import os

class MADGate:
    def __init__(self, 
                 validation_csv,
                 a=None, b=None, 
                 save_to=None
    ):
        self.a = a
        self.b = b
        self.tau = self.find_tau(validation_csv, save_to)

    def find_tau(self, validation_csv,save_to=None):
        """tau = mean of the MAD of each specialist column."""

        table = pd.read_csv(validation_csv)


        # unpack scores
        if "scores" in table.columns:
            # the router's own output: one dict per row  ->  one column per specialist
            table = table["scores"].apply(ast.literal_eval).apply(pd.Series)
        else:
            raise ValueError("no scores column in the validation csv")
       
        
        # MAD of each specialist column
        mads = []
        for column in table.columns:
            scores = table[column]
            middle = scores.median()                   # the column's typical (off-topic) score
            distances = (scores - middle).abs()        # how far each score is from the middle
            mads.append(distances.median())            # MAD = the typical distance

        # one ruler: the average of the column MADs
        tau = sum(mads) / len(mads)


        # the table, to check every number
        summary = pd.DataFrame({"middle (Median)": table.median().values, "MAD": mads},
                               index=table.columns)

        print(summary)
        print(f"sum of MADs = {sum(mads):.4f} | specialists = {len(mads)} | tau = {tau:.4f}")
        if save_to:
            summary.loc["tau"] = [None, tau]
            summary.to_csv(save_to)

        return tau

        
if __name__ == "__main__":
    folder = "../../calibration_threshold/results_validation_scores/validation_eval_service_10"
    file = "cross-encoder_Qwen-Qwen3-Reranker-4B.csv"
    gate = MADGate(os.path.join(folder, file), save_to="tau_table.csv")
    print("tau =", gate.tau)


      
    # def cutoff(self, k):
    #     """T*(K) = (p*(K) / 100) * sqrt(K - 1),  with  p*(K) = a * ln(sqrt(K - 1)) + b"""
    #     z_max = math.sqrt(k - 1)
    #     p = self.a * math.log(z_max) + self.b
    #     return p / 100 * z_max

    # def decide(self, scores_dict, fallback="general"):
    #     scores = list(scores_dict.values())
    #     top_name = max(scores_dict, key=scores_dict.get)

    #     gap = max(scores) - sum(scores) / len(scores)   # best score minus the average score
    #     trust = gap / self.tau                          # the gap counted in "wobbles"
    #     cutoff = self.cutoff(len(scores))
    #     passed = trust >= cutoff

    #     return {
    #         "label": top_name if passed else fallback,
    #         "top_candidate": top_name,
    #         "trust": trust,
    #         "tau": self.tau,
    #         "cutoff": cutoff,
    #         "passed": passed,
    #     }