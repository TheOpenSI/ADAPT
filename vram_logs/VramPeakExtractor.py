from pathlib import Path
import pandas as pd


class VramPeakExtractor:

    def __init__(self):

        self.models_map = {
            "vram_bge_large_en_v1.5_10.csv":
                "BAAI/bge-large-en-v1.5",

            "vram_qwen3_embedding_0.6b_10.csv":
                "Qwen/Qwen3-Embedding-0.6B",

            "vram_qwen3_embedding_4b_10.csv":
                "Qwen/Qwen3-Embedding-4B",

            "vram_qwen3_embedding_8b_10.csv":
                "Qwen/Qwen3-Embedding-8B",

            "vram_minilm_l6_v2_10.csv":
                "sentence-transformers/all-MiniLM-L6-v2",

            "vram_bge_reranker_v2_m3_10.csv":
                "BAAI/bge-reranker-v2-m3",

            "vram_qwen3_reranker_0.6b_10.csv":
                "Qwen/Qwen3-Reranker-0.6B",

            "vram_qwen3_reranker_4b_10.csv":
                "Qwen/Qwen3-Reranker-4B",

            "vram_qwen3_reranker_8b_10.csv":
                "Qwen/Qwen3-Reranker-8B",

            "vram_mxbai_rerank_large_v2_10.csv":
                "mixedbread-ai/mxbai-rerank-large-v2",

            "vram_lfm2_colbert_350m_10.csv":
                "LiquidAI/LFM2-ColBERT-350M",

            "vram_colbertv2_10.csv":
                "colbert-ir/colbertv2.0",

            "vram_reason_moderncolbert_10.csv":
                "lightonai/Reason-ModernColBERT",

            "vram_mxbai_colbert_large_v1_10.csv":
                "mixedbread-ai/mxbai-colbert-large-v1",

            "vram_llama3.1_8b_10.csv":
                "llama3.1:8b",

            "vram_phi4_14b_10.csv":
                "phi4:14b",

            "vram_qwen2.5_7b_10.csv":
                "qwen2.5:7b",

            "vram_qwen2.5_14b_10.csv":
                "qwen2.5:14b",

            "vram_qwen2.5_32b_10.csv":
                "qwen2.5:32b",

            "vram_qwen3_32b_10.csv":
                "qwen3:32b",
        }

        self.vram_results = []


    def _extract_peak(self, filename):

        file = pd.read_csv(
            filename,
            header=None,
            names=["timestamp", "memory_mib"]
        )

        peak_mib = file["memory_mib"].max()

        return peak_mib


    def _extract_all(self):

        self.vram_results = []

        for filename, model_name in self.models_map.items():

            peak_mib = self._extract_peak(filename)

            self.vram_results.append({
                "model_name": model_name,
                "specialist_service_count": 10,
                "peak_vram_mib": peak_mib,
                "peak_vram_gib": round(peak_mib / 1024, 2)
            })

        return self.vram_results


    def _to_dataframe(self):

        return pd.DataFrame(self.vram_results)


    def vram_save_csv(self, output_path="vram_peak_summary.csv"):

        self._extract_all()

        df = self._to_dataframe()

        df.to_csv(
            output_path,
            index=False
        )

        print(f"Saved to: {output_path}")

        return df



if __name__ == "__main__":
    extractor = VramPeakExtractor()
    vram_df = extractor.vram_save_csv()
