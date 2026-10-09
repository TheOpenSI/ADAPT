import os
import sys


sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)

from CalibrationRunner import CalibrationRunner
from TopNCalibrationRunner import TopNCalibrationRunner

from services.CrossEncoderRouter import (
    CrossEncoderRouter
)
# from services.SemanticRouter import SemanticRouter
from services.CrossEncoderRouter import CrossEncoderRouter
# from services.ColBERTRouter import ColBERTRouter
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# models = [
# # "LiquidAI/LFM2-ColBERT-350M",
# # "colbert-ir/colbertv2.0",
# # "lightonai/Reason-ModernColBERT",
# # "mixedbread-ai/mxbai-colbert-large-v1"

# "Qwen/Qwen3-Reranker-4B"

# ]

# for model_name in models:
#     print(f"\nRunning calibration for {model_name}")
    
#     runner = CalibrationRunner(
#         router_cls=ColBERTRouter,

#         model_name=model_name,
        
#         thresholds = list(range(0, 101, 5)),
#         conditions=[
#             3,
#             5,
#             8,
#             10,
#         ],
        
#         data_dir=os.path.join(
#             BASE_DIR,
#             "..",
#             "data"
#         ),
        
#         results_dir=os.path.join(
#             BASE_DIR,
#             "results_auto_threshold"
#         ),
        
#         verbose=False,
#     )

#     runner.run()



# TOPN calibration

# the ~100 off-topic reference queries (the shield uses them to build its cutoff)
ref_path = os.path.join(BASE_DIR, "..", "services", "auto_calibration", "reference_queries.csv")
_ref = pd.read_csv(ref_path)
REFERENCE_QUERIES = _ref["query" if "query" in _ref.columns else _ref.columns[0]].tolist()

# runner = TopNCalibrationRunner(
#     router_cls=CrossEncoderRouter,
#     model_name="Qwen/Qwen3-Reranker-4B",
#     conditions=[8],
#     data_dir=os.path.join(BASE_DIR, "..", "data"),
#     results_dir=os.path.join(BASE_DIR, "results_auto_threshold", "top_n_calibration"),
#     reference_queries=REFERENCE_QUERIES,
#     percentile=10,
# )
# runner.run()

for p in [5, 10, 20, 30]:
    print(f"\n######## PERCENTILE {p} ########")
    runner = TopNCalibrationRunner(
        router_cls=CrossEncoderRouter,
        model_name="Qwen/Qwen3-Reranker-4B",
        conditions=[8],
        data_dir=os.path.join(BASE_DIR, "..", "data"),
        results_dir=os.path.join(BASE_DIR, "results_auto_threshold", "top_n_calibration", f"percentile_{p}"),
        reference_queries=REFERENCE_QUERIES,
        percentile=p,
    )
    runner.run()

