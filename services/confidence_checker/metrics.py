import numpy as np


class ConfidenceMetrics:

    def __init__(self, scores_dict):
        self.scores_dict = scores_dict
        self.values = list(scores_dict.values())

        if len(self.values) < 2:
            raise ValueError("At least 2 candidate scores are required.")

    def max_score(self):
        return float(max(self.values))

    def margin(self):
        sorted_scores = sorted(self.values, reverse=True)

        top1 = sorted_scores[0]
        top2 = sorted_scores[1]

        return float(top1 - top2)

    def std(self):
        return float(np.std(self.values))

    def calculate(self):
        return {
            "k": len(self.values),
            "max": round(self.max_score(), 4),
            "margin": round(self.margin(), 4),
            "std": round(self.std(), 4)
        }


# if __name__ == "__main__":
#     scores = {
#     "anatomy": -3.2,
#     "astronomy": -9.1,
#     "chemistry": -10.4
# }

# confidence = ConfidenceMetrics(scores)

# print(confidence.calculate())