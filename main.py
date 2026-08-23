import os
import sys
import gc
import time
import torch



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


from data_collector.BenchmarkRunner import BenchmarkRunner
from services.SemanticRouter import SemanticRouter
from services.CrossEncoderRouter import CrossEncoderRouter
from services.ColBERTRouter import ColBERTRouter
from services.QueryAnalyser import QueryAnalyser


DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

VERBOSE = True

RUN_CONFIGS = [
    dict(
        cls=SemanticRouter,
        model_name="Qwen/Qwen3-Embedding-8B",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=SemanticRouter,
        model_name="BAAI/bge-large-en-v1.5",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),
    dict(
            cls=SemanticRouter,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),


    dict(
        cls=CrossEncoderRouter,
        model_name="Qwen/Qwen3-Reranker-8B",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=CrossEncoderRouter,
        model_name="mixedbread-ai/mxbai-rerank-large-v2",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=CrossEncoderRouter,
        model_name="BAAI/bge-reranker-v2-m3",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),

    dict(
        cls=ColBERTRouter,
        model_name="lightonai/Reason-ModernColBERT",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=ColBERTRouter,
        model_name="colbert-ir/colbertv2.0",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=ColBERTRouter,
        model_name="mixedbread-ai/mxbai-colbert-large-v1",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=ColBERTRouter,
        model_name="LiquidAI/LFM2-ColBERT-350M",
        service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    ),

    dict(
            cls=QueryAnalyser,
            model_name="qwen3:32b",
            service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=QueryAnalyser,
        model_name="qwen2.5:7b",
        service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=QueryAnalyser,
        model_name="qwen2.5:14b",
        service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    ),
    dict(
        cls=QueryAnalyser,
        model_name="qwen2.5:32b",
        service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    ),

    dict(
            cls=QueryAnalyser,
            model_name="llama3.1:8b",
            service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    ),
    dict(
            cls=QueryAnalyser,
            model_name="phi4:14b",
            service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    ),

]

DATASET_PATHS = [
    os.path.join(DATA_DIR, "dataset", "test_service_10.csv"),
]


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all():

    total_runs = len(RUN_CONFIGS) * len(DATASET_PATHS)

    run_num = 0
    succeeded = 0

    for dataset_path in DATASET_PATHS:
        for cfg in RUN_CONFIGS:
            run_num += 1
            router_name = cfg["cls"].__name__

            print(f"\n{'#' * 70}")
            print(f"# RUN {run_num}/{total_runs}: {router_name} ({cfg['model_name']}) on {dataset_path}")
            print(f"{'#' * 70}\n")

            run_start = time.perf_counter()

            router = None
            runner = None

            try:
                router = cfg["cls"](
                    service_desc_path=cfg["service_desc_path"],
                    model_name=cfg["model_name"],
                    verbose=VERBOSE,
                )

                runner = BenchmarkRunner(dataset_path, 
                                         router, 
                                         results_dir=RESULTS_DIR)
                
                runner.run_test()

                succeeded += 1

            except Exception as e:
                print(
                    f"\n[run_benchmark] FAILED: "
                    f"{router_name} on {dataset_path} — {e}")
                print("[run_benchmark] Skipping to next run.\n")

            finally:
                try:
                    if (router is not None and hasattr(router, "unload_model")):
                        router.unload_model()
                except Exception as e:
                    print(f"[cleanup] Model unload warning: {e}")

                runner = None
                router = None

                cleanup_gpu()

            run_time_s = time.perf_counter() - run_start
            print(f"\n[run_benchmark] {router_name} finished in {run_time_s:.1f}s "
                  f"({run_time_s/60:.1f} min)")

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK COMPLETE — {succeeded}/{total_runs} runs succeeded")
    print(f"{'=' * 70}\n")

if __name__ == "__main__":
    run_all()