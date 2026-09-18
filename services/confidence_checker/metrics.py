import numpy as np

class ConfidenceMetrics:

    def __init__(self, scores_dict):
        self.scores_dict = scores_dict
        self.values = list(scores_dict.values())

        if len(self.values) < 2:
            raise ValueError(
                "At least 2 candidate scores are required."
            )

    def _sorted_scores(self):
        return sorted(self.values, reverse=True)
    
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


    def _score_range(self):
        return float(max(self.values) - min(self.values))


    def _margin_ratio(self):
        """
        margin ratio = top1 - top2 / abs(top1)
        """
        scores = self._sorted_scores()
        top1 = scores[0]
        top2 = scores[1]

        denom = abs(top1)

        if denom < 1e-8:
            return 0.0

        return float((top1 - top2) / denom)


    def _normalized_margin(self):
        """
        normalized_margin = (top1 - top2) / (max_score - min_score)
        """

        score_range = self._score_range()

        if score_range < 1e-8:
            return 0.0

        return float(self._margin() / score_range)


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


    def _second_top_zscore(self):
        zscores = sorted(self._zscores().values(), reverse=True)
        return float(zscores[1])


    def _zscore_margin(self):
        zscores = sorted(self._zscores().values(),reverse=True)
        return float(zscores[0] - zscores[1])


    def calculate(self):

        return {
            "k": len(self.values),

            "max": round(self._max_score(), 4),
            "margin": round(self._margin(), 4),
            "margin_ratio": round(self._margin_ratio(),4),
            "normalized_margin": round(self._normalized_margin(),4),

            "mean": round(self._mean(), 4),
            "std": round(self._std(), 4),

            "zscores": self._zscores(),
            "top_zscore": self._top_zscore(),
            "second_top_zscore": self._second_top_zscore(),

            "zscore_margin": round(self._zscore_margin(),4),

        }


# if __name__ == "__main__":

#     scores = {
#         "anatomy": -3.2,
#         "astronomy": -9.1,
#         "chemistry": -10.4
#     }

#     confidence = ConfidenceMetrics(scores)

#     print(confidence.calculate(c=CONFIDENCE_LEVELS["standard"]))