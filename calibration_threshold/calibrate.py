import os
import sys


sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)

from CalibrationRunner import CalibrationRunner

from services.CrossEncoderRouter import (
    CrossEncoderRouter
)
from services.SemanticRouter import SemanticRouter


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

runner = CalibrationRunner(
    router_cls=SemanticRouter,

    model_name="BAAI/bge-large-en-v1.5",

    thresholds=[
        1.0,
        1.1,
        1.2,
        1.3,
        1.4,
        1.5,
    ],

    conditions=[
        3,
        5,
        8,
        10,
    ],

    data_dir=os.path.join(
        BASE_DIR,
        "..",
        "data"
    ),

    results_dir=os.path.join(
        BASE_DIR,
        "results"
    ),

    verbose=False,
)

runner.run()