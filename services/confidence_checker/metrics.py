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

    def mean(self):
        return float(np.mean(self.values))

    def std(self):
        return float(np.std(self.values))

    def zscores(self):
        """Z-score for every service in this query."""
        std = self.std()
        mean = self.mean()

        if std == 0:
            return {name: 0.0 for name in self.scores_dict}

        return {
            name: round(float((score - mean) / std), 4)
            for name, score in self.scores_dict.items()
        }

    def top_zscore(self):
        """Z-score of just the top-scoring service."""
        z = self.zscores()
        top_name = max(self.scores_dict, key=self.scores_dict.get)
        return z[top_name]

    def confidence_flags(self, c: float = 1.5):
        """True/False per service — does its z-score clear c?"""
        z = self.zscores()
        return {name: bool(val >= c) for name, val in z.items()}

    def is_confident(self, c: float = 1.5) -> bool:
        return self.top_zscore() >= c

    def calculate(self, c: float = 1.5):
        return {
            "k": len(self.values),
            "max": round(self.max_score(), 4),
            "margin": round(self.margin(), 4),
            "mean": round(self.mean(), 4),
            "std": round(self.std(), 4),
            "zscores": self.zscores(),
            "top_zscore": self.top_zscore(),
            "confidence_flags": self.confidence_flags(c),
            "is_confident": self.is_confident(c),
            "c_threshold": c
        }


# if __name__ == "__main__":
#     scores = {
#         "anatomy": -3.2,
#         "astronomy": -9.1,
#         "chemistry": -10.4
#     }

#     confidence = ConfidenceMetrics(scores)
#     print(confidence.calculate())