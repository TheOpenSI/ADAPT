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
# from services.ColBERTRouter import ColBERTRouter
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = [
# # "LiquidAI/LFM2-ColBERT-350M",
# # "colbert-ir/colbertv2.0",
# # "lightonai/Reason-ModernColBERT",
# # "mixedbread-ai/mxbai-colbert-large-v1"

# "Qwen/Qwen3-Reranker-4B"

"BAAI/bge-large-en-v1.5",
]

for model_name in models:
    print(f"\nRunning calibration for {model_name}")
    
    runner = CalibrationRunner(
    router_cls=SemanticRouter,
    model_name=model_name,
    thresholds=[0],                            
    conditions=[10],                           
    data_dir=os.path.join(BASE_DIR, "..", "data"),
    results_dir=os.path.join(BASE_DIR, "results_off_topic_cal", "mmlu"),
    verbose=False,
    )
    runner.run()

