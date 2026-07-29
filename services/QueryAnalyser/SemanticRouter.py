import glob
import json
import numpy as np
import pandas as pd 

import os
import sys 
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from pathlib import Path
from sentence_transformers import SentenceTransformer
from collections import defaultdict



class SemanticRouter:
    def __init__(self):
        self.services_desc_path = "../../data/services_desc.json"
        self.model_name = "MiniLM-L6-v2"
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") 
        self.services_desc = self._get_services_description()

        self.service_names = list(self.services_desc.keys())
        self.services_text = [f"{name}:{desc}" for name, desc in self.services_desc.items()]
        self.services_text_embedding = self.model.encode(
            self.services_text,
            normalize_embeddings=True
            )


    def _get_services_description(self):
        description_file_path = Path(self.services_desc_path).resolve()
        # print(description_file_path)
        
        if not description_file_path.exists():
            raise FileNotFoundError(f"Services description file '{self.services_desc_path}' not found.")
        
        with open(description_file_path, "r") as file:
            data = json.load(file)
            return data

    def _cosine_similarity(self, prompt_embedding, services_text_embedding):
        numerator = services_text_embedding @ prompt_embedding
        # print(numerator)
        query_length = np.linalg.norm(prompt_embedding)
        # print("query_length",query_length)
        service_lengths = np.linalg.norm(services_text_embedding, axis=1)
        # print("service_lengths",service_lengths)
        denominator = service_lengths * query_length
        # print("denominator",denominator)
    
        scores = numerator / denominator
        print("scores", scores)
        best_index = scores.argmax()
        predicted_service = self.service_names[best_index]
        
        return predicted_service

    def route (self, prompt):
        prompt_embedding = self.model.encode(prompt, normalize_embeddings=True)
        return self._cosine_similarity(prompt_embedding, self.services_text_embedding)


    def _run_semantic_inference(self, file_name: str, df: pd.DataFrame):
        result = []
        correct_count = 0
 
        output_file = f"{self.model_name}_{file_name}_query_analysis_results.json"
        output_path = Path(self.services_desc_path).parent.resolve() / "results" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
 
        for _, row in df.iterrows():
            user_query = row['prompt']
            true_label = row['expected_service']
            predicted_service = self.route(user_query)
            response = f"service {predicted_service}"
            is_correct = True if true_label in response else False
 
            if is_correct:
                correct_count += 1
 
            result.append({
                "question": user_query,
                "true_label": true_label,
                "predicted_label": response,
                "is_correct": is_correct
            })
 
            with open(output_path, "w") as f:
                json.dump(result, f, indent=4)
 
        accuracy = correct_count / len(df) if len(df) > 0 else 0
        print(f"Model: {self.model_name} | Dataset: {file_name} | Accuracy: {accuracy:.2%} ({correct_count}/{len(df)})")
 
        return accuracy
    
    
    def run_test(self, data_path="../../data"):
        data_paths = glob.glob(f"{data_path}/*.csv")
        same_model_all_df_results = defaultdict(dict)
 
        for path in data_paths:
            file_name = Path(path).stem
            print(f"Running test for {file_name}...")
            df = pd.read_csv(path)
            accuracy = self._run_semantic_inference(file_name, df)
            same_model_all_df_results[self.model_name][file_name] = accuracy
 
        return same_model_all_df_results


if __name__ == "__main__":
    semantic_test = SemanticRouter()
    semantic_test.run_test()
