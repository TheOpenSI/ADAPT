from abc import ABC, abstractmethod
from pathlib import Path
import json
import numpy as np

class RouterBase(ABC):

    def __init__(self,
                 service_desc_path,
                 model_name
                 
    ): 


        self.service_desc_path = service_desc_path
        self.model_name = model_name
        self.services = self._get_service_descriptions()

        

    def _get_service_descriptions(self):
    # change service_desc_path from string to path
        path = Path(self.service_desc_path)
        if not path.exists():
            raise FileNotFoundError(f"Service description file '{self.service_desc_path}' not found.")
        with open(path, "r") as f:
            data = json.load(f)
            return data


    @staticmethod
    def _calculate_winner_strength(top_zscore,
                                   service_count,
                                   ):
            """
            Calculate the automatically fitted K-aware threshold. 
            
            z_max(K) = sqrt(K - 1)
            
            p*(K) = a * ln(z_max(K)) + b
            
            T*(K) = [p*(K) / 100] * z_max(K)
            
            The winner passes when:
                top_zscore >= T
            """

            # Theoretical maximum top z-score.
            z_max = np.sqrt(service_count - 1)

            # Automatically fitted required percentage.
            # p*(K) = a * ln(sqrt(K - 1)) + b
            required_strength_percentage = (-46.0546 * np.log(np.sqrt(service_count - 1)) + 109.5723)

            # Convert the percentage to the actual z-score threshold.
            automatic_threshold = (required_strength_percentage / 100.0) * z_max

            # Compare the observed top z-score with the threshold.
            strength_passed = (float(top_zscore) >= automatic_threshold)
            
            # Observed winner strength relative to z_max.
            winner_strength_ratio = (float(top_zscore)/ z_max)
            winner_strength_percentage = (winner_strength_ratio * 100.0)
    
            return {
                "service_count": int(service_count),
                "top_zscore": float(top_zscore),
                "z_max": float(z_max),
                "winner_strength_ratio": float(winner_strength_ratio),
                "winner_strength_percentage": float(winner_strength_percentage),
                "required_strength_percentage": float(required_strength_percentage),
                "automatic_threshold": float(automatic_threshold),
                "strength_passed": bool(strength_passed),
            }

    @abstractmethod
    def route(self, query):
        pass



