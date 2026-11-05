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
RESULTS_DIR = os.path.join(BASE_DIR, "medmcqa_mmlu_as_general") 

VERBOSE = False

APPLY_GATE = True 
GATE = "gap"    # "gap" or "margin" or "max"
FORM = "log" 

# CONDITIONS = [3,5,8,10,12,14]
CONDITIONS = [3,5,7,9]


MODEL_CONFIGS = [

    # MMLU 

    # dict(cls=SemanticRouter,     model_name="Qwen/Qwen3-Embedding-0.6B", gap_c0=0.0760, gap_c1=0.0264),
    # dict(cls=SemanticRouter,     model_name="Qwen/Qwen3-Embedding-4B",   gap_c0=0.0512, gap_c1=0.0385),
    # dict(cls=SemanticRouter,     model_name="Qwen/Qwen3-Embedding-8B",   gap_c0=0.0537, gap_c1=0.0325),
    # dict(cls=SemanticRouter,     model_name="sentence-transformers/all-MiniLM-L6-v2", gap_c0=0.0580, gap_c1=0.0333),
    # # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-0.6B",  gap_c0=0.6296, gap_c1=0.4987),
    # # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-8B",    gap_c0=0.5141, gap_c1=0.6713),
    # # dict(cls=CrossEncoderRouter, model_name="BAAI/bge-reranker-v2-m3",   gap_c0=-0.00001493, gap_c1=0.00006167),
    # dict(cls=ColBERTRouter,      model_name="colbert-ir/colbertv2.0",    gap_c0=0.0369, gap_c1=0.0236),
    # dict(cls=ColBERTRouter,      model_name="mixedbread-ai/mxbai-colbert-large-v1", gap_c0=0.0771, gap_c1=0.0215),
    
    # dict(cls=SemanticRouter, model_name="BAAI/bge-large-en-v1.5", gap_c0=0.0436, gap_c1=0.0163), 
    # # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B", gap_c0=0.9332, gap_c1=0.3971),
    # # dict(cls=CrossEncoderRouter, model_name="mixedbread-ai/mxbai-rerank-large-v2", gap_c0=-0.0994, gap_c1=0.1976),  
    # dict(cls=ColBERTRouter, model_name="lightonai/Reason-ModernColBERT", gap_c0=0.009360, gap_c1=0.0134), 
    # dict(cls=ColBERTRouter, model_name="LiquidAI/LFM2-ColBERT-350M", gap_c0=0.0003873, gap_c1=0.0005357),
    
    
    # MEDMCQA
    # dict(cls=SemanticRouter,     model_name="Qwen/Qwen3-Embedding-0.6B", gap_c0=0.057401, gap_c1=0.032875),
    # dict(cls=SemanticRouter,     model_name="Qwen/Qwen3-Embedding-4B",   gap_c0=0.053238, gap_c1=0.019881),
    dict(cls=SemanticRouter,     model_name="Qwen/Qwen3-Embedding-8B",   gap_c0=0.032083, gap_c1=0.043901),
    # dict(cls=SemanticRouter,     model_name="sentence-transformers/all-MiniLM-L6-v2", gap_c0=0.058811, gap_c1=0.032671),
    # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-0.6B",  gap_c0=0.291404, gap_c1=0.703123),
    dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-8B",    gap_c0=0.932232, gap_c1=0.277994),
    # dict(cls=CrossEncoderRouter, model_name="BAAI/bge-reranker-v2-m3",   gap_c0=-0.000038, gap_c1=0.000095),
    # dict(cls=ColBERTRouter,      model_name="colbert-ir/colbertv2.0",    gap_c0=0.043856, gap_c1=0.024976),
    # dict(cls=ColBERTRouter,      model_name="mixedbread-ai/mxbai-colbert-large-v1", gap_c0=0.063783, gap_c1=0.026992),

    dict(cls=SemanticRouter,     model_name="BAAI/bge-large-en-v1.5",    gap_c0=0.030415, gap_c1=0.020155),
    dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",    gap_c0=0.890826, gap_c1=0.616486),
    dict(cls=CrossEncoderRouter, model_name="mixedbread-ai/mxbai-rerank-large-v2", gap_c0=0.050393, gap_c1=0.074527),
    dict(cls=ColBERTRouter,      model_name="lightonai/Reason-ModernColBERT", gap_c0=0.003726, gap_c1=0.018837),
    dict(cls=ColBERTRouter,      model_name="LiquidAI/LFM2-ColBERT-350M", gap_c0=0.000571, gap_c1=0.000518),


    # dict(cls=SemanticRouter,     model_name="BAAI/bge-large-en-v1.5",         margin_c0=0.08763, margin_c1=-0.02419),
    # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",         margin_c0=1.83264, margin_c1=-0.44286),
    # dict(cls=ColBERTRouter,      model_name="lightonai/Reason-ModernColBERT", margin_c0=0.02531, margin_c1=-0.00174),

    # dict(cls=SemanticRouter,     model_name="BAAI/bge-large-en-v1.5",         max_c0=0.40949,   max_c1=0.01799),
    # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",         max_c0=-10.03722, max_c1=0.87981),
    # dict(cls=ColBERTRouter,      model_name="lightonai/Reason-ModernColBERT", max_c0=0.58796,   max_c1=0.00308),

    # dict(cls=SemanticRouter,     model_name="BAAI/bge-large-en-v1.5",         zscore_c0=0.81821, zscore_c1=0.55536),
    # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",         zscore_c0=0.65949, zscore_c1=0.66648),
    # dict(cls=ColBERTRouter,      model_name="lightonai/Reason-ModernColBERT", zscore_c0=0.48782, zscore_c1=0.83210),


    # # GAP, FORM = "linear"   (T(k) = c0 + c1 * k)
    # dict(cls=SemanticRouter,     model_name="BAAI/bge-large-en-v1.5",         gap_c0=0.05345, gap_c1=0.00312),
    # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",         gap_c0=1.16600, gap_c1=0.07696),
    # dict(cls=ColBERTRouter,      model_name="lightonai/Reason-ModernColBERT", gap_c0=0.01718, gap_c1=0.00260),


    # # GAP, FORM = "sqrt"     (T(k) = c0 + c1 * sqrt(k))
    # dict(cls=SemanticRouter,     model_name="BAAI/bge-large-en-v1.5",         gap_c0=0.03712, gap_c1=0.01445),
    # dict(cls=CrossEncoderRouter, model_name="Qwen/Qwen3-Reranker-4B",         gap_c0=0.76893, gap_c1=0.35433),
    # dict(cls=ColBERTRouter,      model_name="lightonai/Reason-ModernColBERT", gap_c0=0.00380, gap_c1=0.01195),


    # dict(cls=QueryAnalyser, model_name="qwen2.5:14b"),
    # dict(cls=QueryAnalyser, model_name="qwen2.5:32b"),
    # dict(cls=QueryAnalyser, model_name="phi4:14b"),
    # dict(cls=QueryAnalyser, model_name="qwen2.5:7b"),
    # dict(cls=QueryAnalyser, model_name="llama3.1:8b"),
    # dict(cls=QueryAnalyser, model_name="qwen3:32b"),
]



def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all():

    total_runs = len(MODEL_CONFIGS) * len(CONDITIONS)

    run_num = 0
    succeeded = 0

    for k in CONDITIONS:
        service_desc_path = os.path.join(DATA_DIR, "dataset", f"medmcqa_service_descriptions_{k}.json")
        dataset_path = os.path.join(DATA_DIR, "dataset", f"medmcqa_test_eval_service_{k}_mmlu_general.csv")
        # dataset_path = os.path.join(DATA_DIR, "dataset", f"mmlu_test_eval_service_{k}_300.csv")
        # dataset_path = os.path.join(DATA_DIR, "dataset", f"mmlu_test_eval_service_{k}_prop.csv")
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
                if cfg["cls"] is QueryAnalyser:
                    # LLM router: no gate
                    router = QueryAnalyser(
                        service_desc_path=service_desc_path,
                        model_name=cfg["model_name"],
                        verbose=VERBOSE,
                    )
                else:
                    router = cfg["cls"](
                        service_desc_path=service_desc_path,
                        model_name=cfg["model_name"],
                        verbose=VERBOSE,
                        apply_threshold=APPLY_GATE,
                        gate=GATE,
                        form=FORM,
                        gap_c0=cfg.get("gap_c0"),
                        gap_c1=cfg.get("gap_c1"),
                        margin_c0=cfg.get("margin_c0"),
                        margin_c1=cfg.get("margin_c1"),
                        max_c0=cfg.get("max_c0"),
                        max_c1=cfg.get("max_c1"),
                        zscore_c0=cfg.get("zscore_c0"),
                        zscore_c1=cfg.get("zscore_c1"),
                        fixed_threshold=cfg.get("fixed_threshold"),
                )

                runner = BenchmarkRunner(dataset_path, 
                                         router, 
                                         results_dir=RESULTS_DIR)
                
                runner.run_test()
                print("\nBENCHMARK FINISHED")

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