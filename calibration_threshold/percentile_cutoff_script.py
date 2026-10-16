import os
import sys
import traceback

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from calibration_threshold.PercentileGateSweepRunner import PercentileGateSweepRunner
from services.SemanticRouter import SemanticRouter
from services.CrossEncoderRouter import CrossEncoderRouter
from services.ColBERTRouter import ColBERTRouter    

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

models = [
    (SemanticRouter, "BAAI/bge-large-en-v1.5",
        {85: (0.03595683599, 0.01733355608), 90: (0.04362318174, 0.01632105895), 95: (0.06034572118, 0.01137800542), 99: (0.07249414986, 0.01386018132)}),
    (SemanticRouter, "Qwen/Qwen3-Embedding-0.6B",
        {85: (0.06596884529, 0.02801711009), 90: (0.07603420015, 0.0263733064), 95: (0.09912628519, 0.01828226236), 99: (0.1086535038, 0.02796803186)}),
    (SemanticRouter, "Qwen/Qwen3-Embedding-4B",
        {85: (0.04006061627, 0.04112001258), 90: (0.05120974864, 0.03847158405), 95: (0.06803141093, 0.03342592637), 99: (0.09392125415, 0.02569386401)}),
    (SemanticRouter, "Qwen/Qwen3-Embedding-8B",
        {85: (0.04740239371, 0.03272187899), 90: (0.05373225902, 0.03252864384), 95: (0.0636686776, 0.03156652361), 99: (0.08266001977, 0.02805026298)}),
    (SemanticRouter, "sentence-transformers/all-MiniLM-L6-v2",
        {85: (0.04518712245, 0.03352811891), 90: (0.05801207688, 0.03330132779), 95: (0.06311894523, 0.04481924383), 99: (0.1156100743, 0.03713984983)}),
    (ColBERTRouter, "LiquidAI/LFM2-ColBERT-350M",
        {85: (0.0002323555294, 0.000532205711), 90: (0.0003873312792, 0.0005357293237), 95: (0.0004227669961, 0.0007384755849), 99: (0.001176410548, 0.0009553639405)}),
    (ColBERTRouter, "colbert-ir/colbertv2.0",
        {85: (0.02082844094, 0.02763416118), 90: (0.03688833165, 0.02356116877), 95: (0.04138555792, 0.03286578245), 99: (0.08264049893, 0.03427615185)}),
    (ColBERTRouter, "lightonai/Reason-ModernColBERT",
        {85: (0.008137688416, 0.01118384261), 90: (0.00935986829, 0.01338595543), 95: (0.01500081154, 0.01468717698), 99: (0.03456384535, 0.01121885535)}),
    (ColBERTRouter, "mixedbread-ai/mxbai-colbert-large-v1",
        {85: (0.05927744866, 0.02519531102), 90: (0.07712228083, 0.02146292499), 95: (0.07046016465, 0.04114706649), 99: (0.1113462064, 0.02946247386)}),
    (CrossEncoderRouter, "BAAI/bge-reranker-v2-m3",
        {85: (-5.01492903e-05, 5.106628971e-05), 90: (-1.493189307e-05, 6.166784304e-05), 95: (-2.693861365e-05, 0.0001402592835), 99: (-0.0009711589343, 0.002061601172)}),
    (CrossEncoderRouter, "Qwen/Qwen3-Reranker-0.6B",
        {85: (0.4039552085, 0.5111365132), 90: (0.629623299, 0.4986952531), 95: (0.7944647184, 0.6484455962), 99: (1.452614336, 0.7360058985)}),
    (CrossEncoderRouter, "Qwen/Qwen3-Reranker-4B",
        {85: (0.8460206135, 0.3479670065), 90: (0.9332465447, 0.3971455789), 95: (1.081622535, 0.4942510031), 99: (1.41772783, 0.759478451)}),
    (CrossEncoderRouter, "Qwen/Qwen3-Reranker-8B",
        {85: (0.2982952017, 0.6602107432), 90: (0.5140942866, 0.6713010306), 95: (0.7423616131, 0.815829616), 99: (0.9851008853, 1.270030735)}),
    (CrossEncoderRouter, "mixedbread-ai/mxbai-rerank-large-v2",
        {85: (-0.0314644161, 0.1012197565), 90: (-0.09939555255, 0.1976338995), 95: (-0.01317489272, 0.2246693605), 99: (0.2780964804, 0.1250417952)}),
]

failed = []
for router_cls, model_name, constants in models:
    print(f"\nRunning percentile sweep for {model_name}")
    try:
        runner = PercentileGateSweepRunner(
            router_cls=router_cls,
            model_name=model_name,
            conditions=[3,5,8,10],
            data_dir=os.path.join(BASE_DIR, "..", "data"),
            results_dir=os.path.join(BASE_DIR, "results_validation_scores"),
            constants=constants,
            verbose=False,
        )
        runner.run()
    except Exception:
        traceback.print_exc()
        failed.append(model_name)

print("\nFailed models:", failed or "none")