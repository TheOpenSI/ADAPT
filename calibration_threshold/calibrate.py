import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)

from CalibrationRunner import CalibrationRunner

from services.CrossEncoderRouter import (
    CrossEncoderRouter
)

from services.confidence_checker.config import (
    CONFIDENCE_LEVELS
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

runner = CalibrationRunner(
    router_cls=CrossEncoderRouter,

    model_name="Qwen/Qwen3-Reranker-4B",

    thresholds=[
        1
    ],

    conditions=[
        5,8,10
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