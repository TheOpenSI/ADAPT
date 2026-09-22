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
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.90,
        1.00,
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
        "results_hybrid"
    ),

    verbose=False,
)

runner.run()