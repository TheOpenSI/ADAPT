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


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = [
"BAAI/bge-reranker-v2-m3",
"Qwen/Qwen3-Reranker-0.6B",
"Qwen/Qwen3-Reranker-8B",
# "mixedbread-ai/mxbai-rerank-large-v2",
]


for model_name in models:
    print(f"\nRunning calibration for {model_name}")
    
    runner = CalibrationRunner(
        router_cls=CrossEncoderRouter,

        model_name=model_name,
        
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