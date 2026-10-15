import os
import sys
import gc
import time
import torch
import pandas as pd
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


from data_collector.BenchmarkRunner import BenchmarkRunner
from services.SemanticRouter import SemanticRouter
from services.CrossEncoderRouter import CrossEncoderRouter
from services.ColBERTRouter import ColBERTRouter
from services.QueryAnalyser import QueryAnalyser
from services.TauNoteBook import TauNotebook


DATA_DIR = os.path.join(BASE_DIR, "data")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_vram")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_per_K_threshold")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_global_threshold")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_mmlu")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_medmcqa")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_massive")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_mmlu_validation_fixed")

# RESULTS_DIR = os.path.join(BASE_DIR, "results_strength_main", "balanced")
# RESULTS_DIR = os.path.join(BASE_DIR, "results_gap_medmcqa")  
RESULTS_DIR = os.path.join(BASE_DIR, "results_300")  

VERBOSE = False

GATE_MODE = "gap"   

# CONDITIONS = [3,5,7,9]
CONDITIONS = [3,5,8,10,12,14]
# CONDITIONS = [8]

MODEL_CONFIGS = [
    # dict(cls=SemanticRouter, model_name="Qwen/Qwen3-Embedding-0.6B"),
    # dict(cls=SemanticRouter, model_name="Qwen/Qwen3-Embedding-4B"),
    # dict(cls=SemanticRouter, model_name="Qwen/Qwen3-Embedding-8B"),
    # dict(cls=SemanticRouter, model_name="BAAI/bge-large-en-v1.5",
    #  gap_c0=0.0436, gap_c1=0.0163), 
    # # dict(cls=SemanticRouter, model_name="sentence-transformers/all-MiniLM-L6-v2"),
    # # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-0.6B"),
    # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",
    #      gap_c0=0.9332, gap_c1=0.3971),
    # # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-8B"),
    # dict(cls=CrossEncoderRouter, model_name="mixedbread-ai/mxbai-rerank-large-v2",
    #      gap_c0=-0.0994, gap_c1=0.1976),  

    # dict(cls=ColBERTRouter, model_name="lightonai/Reason-ModernColBERT",
    #  gap_c0=0.009360, gap_c1=0.013386),
     
    # dict(cls=ColBERTRouter, model_name="LiquidAI/LFM2-ColBERT-350M",
    #  gap_c0=0.000387, gap_c1=0.000536),
    # dict(cls=CrossEncoderRouter, model_name="BAAI/bge-reranker-v2-m3"),
    # dict(cls=ColBERTRouter, model_name="lightonai/Reason-ModernColBERT"),
    # dict(cls=ColBERTRouter, model_name="colbert-ir/colbertv2.0"),
    # dict(cls=ColBERTRouter, model_name="mixedbread-ai/mxbai-colbert-large-v1"),
    # dict(cls=ColBERTRouter, model_name="LiquidAI/LFM2-ColBERT-350M"),


    dict(cls=QueryAnalyser, model_name="qwen2.5:14b"),
    # dict(cls=QueryAnalyser, model_name="qwen2.5:32b"),
    dict(cls=QueryAnalyser, model_name="phi4:14b"),
    # dict(cls=QueryAnalyser, model_name="qwen2.5:7b"),
    dict(cls=QueryAnalyser, model_name="llama3.1:8b"),
    # dict(cls=QueryAnalyser, model_name="qwen3:32b"),
]


#     # MAD
#     dict(cls=SemanticRouter, model_name="BAAI/bge-large-en-v1.5",
#              c0=0.6659, c1=0.8367),
#     dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",
#              c0=-0.2544, c1=1.0414),
#     dict(cls=CrossEncoderRouter, model_name="mixedbread-ai/mxbai-rerank-large-v2",
#                       c0=-10.042, c1=17.958),


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all():

    total_runs = len(MODEL_CONFIGS) * len(CONDITIONS)

    run_num = 0
    succeeded = 0

    for k in CONDITIONS:
        # service_desc_path = os.path.join(DATA_DIR, "dataset", f"medmcqa_service_descriptions_{k}.json")
        # service_desc_path = os.path.join(DATA_DIR, "dataset", f"massive_service_descriptions_{k}.json")
        service_desc_path = os.path.join(DATA_DIR, "dataset", f"service_descriptions_{k}.json")

        # MUST CHANGE DEV to TEST 
        # dataset_path = os.path.join(DATA_DIR, "dataset", f"medmcqa_test_eval_service_{k}.csv")
        # dataset_path = os.path.join(DATA_DIR, "dataset", f"massive_test_eval_service_{k}.csv")
        dataset_path = os.path.join(DATA_DIR, "dataset", f"mmlu_test_eval_service_{k}_300.csv")
        # dataset_path = os.path.join(DATA_DIR, "dataset", f"test_eval_service_{k}.csv")
        # dataset_path = os.path.join(DATA_DIR, "dataset", f"mmlu_validation_eval_service_3_fixed.csv")
        # dataset_path = os.path.join(DATA_DIR, "dataset", f"dummy_general.csv")
        

        print(f"\n{'*' * 70}")
        print(f"* CONDITION k={k}")
        print(f"*   service_desc_path: {service_desc_path}")
        print(f"*   dataset_path:      {dataset_path}")
        print(f"{'*' * 70}")


        for cfg in MODEL_CONFIGS:
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
                    service_desc_path=service_desc_path,
                    model_name=cfg["model_name"],
                    verbose=VERBOSE,
                    # gate_mode=GATE_MODE,
                    # gap_c0=cfg["gap_c0"], 
                    # gap_c1=cfg["gap_c1"], 
                    # a=cfg["a"],
                    # b=cfg["b"],
                )

                # print("\nMODEL LOADED AND READY")
                # input("Start nvidia-smi logging now, then press Enter to begin benchmark...")

                if GATE_MODE == "mad":
                    notebook = TauNotebook(
                        os.path.join(BASE_DIR, "calibration_threshold", "dataset", "dummy_general.csv"),
                        os.path.join(BASE_DIR, "calibration_threshold", "results_cut_off",
                                     "mmlu_" + cfg["model_name"].replace("/", "_") + ".csv"),
                    )

                    def scorer(questions, name):
                        description = router.services[name]
                        if isinstance(router, SemanticRouter):
                            q_vecs = router.model.encode(questions)
                            d_vec = router.model.encode(description)
                            return [router._cosine_similarity(q_vec, d_vec) for q_vec in q_vecs]
                        return router.score_queries(router.model_name, questions, description)

 
                    names = list(router.services)
                    router.mad_tau = notebook.tau(list(router.services), scorer)
                    router.c0 = cfg["c0"]
                    router.c1 = cfg["c1"]
                    cutoff = router.c0 + router.c1 * math.log(len(names))
                    print(f"[run_benchmark] K={k}  tau={router.mad_tau:.4f} cutoff={cutoff:.4f}")

                runner = BenchmarkRunner(dataset_path, 
                                         router, 
                                         results_dir=RESULTS_DIR)
                
                runner.run_test()
                print("\nBENCHMARK FINISHED")

                # input(
                #     "Stop the nvidia-smi logger now, "
                #     "then press Enter to unload the model..."
                # )

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