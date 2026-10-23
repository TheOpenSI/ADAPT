from abc import ABC, abstractmethod
from pathlib import Path
import json


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

    @abstractmethod
    def route(self, query):
        pass