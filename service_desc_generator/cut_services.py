import json
import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)
# from data.dataset.config  import SERVICE_SETS
from data.ScienceQA.specialists_dataset.scienceqa_config import SCIENCEQA_SERVICE_SETS


def build_service_description_sets(
    frozen_descriptions,
    service_sets
):
    description_sets = {}

    for size, specialists in service_sets.items():

        descriptions = {
            service: frozen_descriptions[service]
            for service in specialists
        }

        # descriptions["general"] = (
        #     frozen_descriptions["general"]
        # )

        description_sets[size] = descriptions

    return description_sets


with open("data/ScienceQA/descriptions/service_descriptions_8.json", 
          "r") as f:
     frozen_descriptions = json.load(f)

description_sets = build_service_description_sets(
    frozen_descriptions,
    SCIENCEQA_SERVICE_SETS
)

import json

description_sets = build_service_description_sets(
    frozen_descriptions,
    SCIENCEQA_SERVICE_SETS
)

for size, descriptions in description_sets.items():

    out_path = f"../data/dataset/scienceqa_service_descriptions_{size}.json"

    with open(out_path, "w") as f:
        json.dump(
            descriptions,
            f,
            indent=2
        )

    print(
        f"[SAVED] {out_path}"
    )


