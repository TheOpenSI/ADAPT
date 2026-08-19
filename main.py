import os
import sys
import gc
import time
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from evaluator.Evaluator import Evaluator
from services.SemanticRouter import SemanticRouter
# from services.CrossEncoderRouter import CrossEncoderRouter
# from services.ColBERTRouter import ColBERTRouter
from services.QueryAnalyser import QueryAnalyser


DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

VERBOSE = True
RESUME = True

RUN_CONFIGS = [
    # dict(
    #     cls=SemanticRouter,
    #     model_name="Qwen/Qwen3-Embedding-8B",
    #     service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    # ),
    # dict(
    #     cls=SemanticRouter,
    #     model_name="BAAI/bge-large-en-v1.5",
    #     service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    # ),
    # dict(
    #         cls=SemanticRouter,
    #         model_name="sentence-transformers/all-MiniLM-L6-v2",
    #         service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    # ),


    # dict(
    #     cls=CrossEncoderRouter,
    #     model_name="Qwen/Qwen3-Reranker-8B",
    #     service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_10.json"),
    # ),


    # dict(
    #     cls=ColBERTRouter,
    #     model_name="colbert-ir/colbertv2.0",
    #     service_desc_path=os.path.join(DATA_DIR, "dataset", "service_descriptions_5.json"),
    # ),

    # dict(
    #         cls=QueryAnalyser,
    #         model_name="qwen3:32b",
    #         service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_5.json"),
    # ),

    # dict(
    #     cls=QueryAnalyser,
    #     model_name="qwen2.5:7b",
    #     service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    # ),
    # dict(
    #     cls=QueryAnalyser,
    #     model_name="qwen2.5:14b",
    #     service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    # ),
    # dict(
    #     cls=QueryAnalyser,
    #     model_name="qwen2.5:32b",
    #     service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    # ),

    # dict(
    #         cls=QueryAnalyser,
    #         model_name="llama3.1:8b",
    #         service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    # ),
    # dict(
    #         cls=QueryAnalyser,
    #         model_name="phi4:14b",
    #         service_desc_path=os.path.join(DATA_DIR,"dataset", "service_descriptions_10.json"),
    # ),

]

DATASET_PATHS = [
    os.path.join(DATA_DIR, "dataset", "test_service_10.csv"),
]


def cleanup_gpu(router):
    """Free VRAM between routers so one method's leftover memory doesn't
    pollute the next method's gpu_memory_gb readings. Safe to call with
    router=None (e.g. if construction itself failed before assignment)."""
    if router is not None:
        del router
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_all():
    all_summaries = []
    total_runs = len(RUN_CONFIGS) * len(DATASET_PATHS)
    run_num = 0

    for dataset_path in DATASET_PATHS:
        for cfg in RUN_CONFIGS:
            run_num += 1
            router_name = cfg["cls"].__name__

            print(f"\n{'#' * 70}")
            print(f"# RUN {run_num}/{total_runs}: {router_name} ({cfg['model_name']}) on {dataset_path}")
            print(f"{'#' * 70}\n")

            run_start = time.perf_counter()
            router = None

            try:
                router = cfg["cls"](
                    service_desc_path=cfg["service_desc_path"],
                    model_name=cfg["model_name"],
                    verbose=VERBOSE,
                )

                evaluator = Evaluator(dataset_path, router, results_dir=RESULTS_DIR,resume=RESUME)
                summary = evaluator.run_test()
                summary["dataset"] = dataset_path
                all_summaries.append(summary)

            except Exception as e:
                print(f"\n[run_benchmark] FAILED: {router_name} on {dataset_path} — {e}")
                print(f"[run_benchmark] Skipping to next run.\n")
                continue

            finally:
                cleanup_gpu(router)
                if hasattr(router, "unload_model"):
                    router.unload_model()

            run_time_s = time.perf_counter() - run_start
            print(f"\n[run_benchmark] {router_name} finished in {run_time_s:.1f}s "
                  f"({run_time_s/60:.1f} min)")

    print(f"\n{'=' * 70}")
    print(f"BENCHMARK COMPLETE — {len(all_summaries)}/{total_runs} runs succeeded")
    print(f"{'=' * 70}\n")

    for s in all_summaries:
        print(f"  {s['routing_method']:15s} | {s['model_name']:35s} | "
              f"acc={s['accuracy']:.3f} | macro_f1={s['macro_f1']:.3f} | "
              f"avg_latency={s['avg_latency_ms']:.1f}ms")

    return all_summaries


if __name__ == "__main__":
    run_all()