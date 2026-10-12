import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # project root
sys.path.append(BASE_DIR)

from services.TauNoteBook import TauNotebook
from services.CrossEncoderRouter import CrossEncoderRouter
from services.SemanticRouter import SemanticRouter
# from services.ColBERTRouter import ColBERTRouter

ROUTER_CLASS = SemanticRouter      
DATASET      = "mmlu"              
DESC_NAME    = "service_descriptions_{k}.json"   
KS           = [3, 5, 8, 10]
PERCENTILE   = 90

models = [
    # "Qwen/Qwen3-Reranker-4B",
    "BAAI/bge-large-en-v1.5",
    # "colbert-ir/colbertv2.0",

    # "mixedbread-ai/mxbai-rerank-large-v2",
]

rows = []
for k in KS:
    for model in models:
        router = ROUTER_CLASS(
            service_desc_path=os.path.join(BASE_DIR, "data", "dataset", DESC_NAME.format(k=k)),
            model_name=model, verbose=False, gate_mode="mad",
        )
        notebook = TauNotebook(
            dummy_general_csv = os.path.join(BASE_DIR, "calibration_threshold", "dataset", "dummy_general.csv"),
            output_csv = os.path.join(BASE_DIR, "calibration_threshold", "results_cut_off",
                              f"{DATASET}_{model.replace('/', '_')}.csv"),
        )

        def scorer(questions, specialist):
            description = router.services[specialist]
            if ROUTER_CLASS is SemanticRouter:
                q_vecs = router.model.encode(questions)         # one vector per question
                d_vec = router.model.encode(description)        # one vector for the description
                return [router._cosine_similarity(q_vec, d_vec) for q_vec in q_vecs]
            return router.score_queries(router.model_name, questions, description)


        specialists = [name for name in router.services] # take name from the router.services and put in a list
        tau = notebook.tau(specialists, scorer)
        cut = notebook.cutoff(specialists, tau, PERCENTILE, show=True)
        rows.append((model, k, tau, cut))

        print(f"\n{'#' * 60}")
        print(f"SUMMARY  model={model}  K={k}")
        print(f"  description file : {DESC_NAME.format(k=k)}")
        print(f"  specialists ({len(specialists)}): {specialists}")
        print(f"  tau              : {tau:.4f}")
        print(f"  cutoff (p{PERCENTILE})    : {cut:.4f}")
        print(f"{'#' * 60}\n")
   

print(f"\nmodel, K, tau, cutoff (percentile {PERCENTILE})")
for model, k, tau, cut in rows:
    print(f"{model}, {k}, {tau:.4f}, {cut:.4f}")

# ---- new: save the table ----
summary = pd.DataFrame(rows, columns=["model", "K", "tau", f"cutoff_p{PERCENTILE}"])

for model in models:
    model_table = summary[summary["model"] == model]
    summary_path = os.path.join(BASE_DIR, "calibration_threshold", "results_cut_off",
                                f"{DATASET}_{model.replace('/', '_')}_cutoff_table_p{PERCENTILE}.csv")
    model_table.to_csv(summary_path, index=False)
    print(f"saved cutoff table for {model} to {summary_path}")