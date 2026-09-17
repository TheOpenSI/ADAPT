import numpy as np
from .config import CONFIDENCE_LEVELS

class ConfidenceMetrics:

    def __init__(self, scores_dict):
        self.scores_dict = scores_dict
        self.values = list(scores_dict.values())

        if len(self.values) < 2:
            raise ValueError(
                "At least 2 candidate scores are required."
            )

    def _max_score(self):
        return float(max(self.values))

    def _margin(self):
        sorted_scores = sorted(
            self.values,
            reverse=True
        )

        top1 = sorted_scores[0]
        top2 = sorted_scores[1]

        return float(top1 - top2)

    def _mean(self):
        return float(np.mean(self.values))

    def _std(self):
        return float(np.std(self.values))

    # Convert all service scores into z-scores
    def _zscores(self):

        std = self._std()
        mean = self._mean()

        # Handle the case where all scores are identical
        if std == 0:
            return {
                name: 0.0
                for name in self.scores_dict
            }

        zscore_dict = {}

        for name, score in self.scores_dict.items():
            z = (score - mean) / std
            zscore_dict[name] = round(
                float(z),
                4
            )

        return zscore_dict

    def _top_zscore(self):

        names = list(self.scores_dict.keys())
        values = list(self.scores_dict.values())

        top_idx = values.index(max(values))
        top_name = names[top_idx]

        # Get the winning service's z-score
        all_zscores = self._zscores()

        return all_zscores[top_name]

    def _confidence_flags(self, c: float):
        """Pass or fail the threshold for each service."""

        z = self._zscores()

        return {
            name: val >= c
            for name, val in z.items()
        }

    def _is_confident(self, c: float):
        """Does the winning service pass the threshold?"""

        return self._top_zscore() >= c

    def calculate(self, c: float):

        return {
            "k": len(self.values),

            "max": round(self._max_score(), 4),
            "margin": round(self._margin(), 4),
            "mean": round(self._mean(), 4),
            "std": round(self._std(), 4),

            "zscores": self._zscores(),
            "top_zscore": self._top_zscore(),

            "confidence_flags": self._confidence_flags(c),
            "is_confident": self._is_confident(c),

            "c_threshold": c,
        }


# if __name__ == "__main__":

#     scores = {
#         "anatomy": -3.2,
#         "astronomy": -9.1,
#         "chemistry": -10.4
#     }

#     confidence = ConfidenceMetrics(scores)

#     print(confidence.calculate(c=CONFIDENCE_LEVELS["standard"]))