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
from services.CrossEncoderRouter import CrossEncoderRouter


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

runner = CalibrationRunner(
    router_cls=CrossEncoderRouter,

    model_name="Qwen/Qwen3-Reranker-4B",

    thresholds = list(range(0, 101, 5)),

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
        "results_auto_threshold"
    ),

    verbose=False,
)

runner.run()