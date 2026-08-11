import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from dataset_builder.config import SERVICE_SETS
from pathlib import Path
import json
import pandas as pd


class ServiceProfileBuilder:
    """
    Build TP/FN/FP routing profiles for each service, from the routing results produced by RoutingTester.

    For a given service:
      TRUE POSITIVES  - expected == service, predicted == service
      FALSE NEGATIVES - expected == service, predicted != service (missed)
      FALSE POSITIVES - expected != service, predicted == service (wrongly attracted)
    """

    def __init__(self, services, results_dir="../data"):
        self.services = services
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _true_positives(self, df, service):
        mask = (df["expected_service"] == service) & (df["predicted_service"] == service)
        return df[mask]

    def _false_negatives(self, df, service):
        mask = (df["expected_service"] == service) & (df["predicted_service"] != service)
        return df[mask]

    def _false_positives(self, df, service):
        mask = (df["expected_service"] != service) & (df["predicted_service"] == service)
        return df[mask]

    def _build_service_profile(self, df, service):

        service_rows = df[df["expected_service"] == service]

        true_positives = self._true_positives(df, service)
        false_negatives = self._false_negatives(df, service)
        false_positives = self._false_positives(df, service)

        print(f"    {service}: TP={len(true_positives)} FN={len(false_negatives)} FP={len(false_positives)}")

        true_positive_examples = []
        for _, row in true_positives.iterrows():
            print(f"      [TP] {row['question'][:60]}...")
            true_positive_examples.append({
                "question": row["question"],
                "choices": row["choices"],
            })

        false_negative_examples = []
        for _, row in false_negatives.iterrows():
            print(f"      [FN] {row['question'][:60]}... -> routed_to={row['predicted_service']}")
            false_negative_examples.append({
                "question": row["question"],
                "choices": row["choices"],
                "routed_to": row["predicted_service"],
            })

        false_positive_examples = []
        for _, row in false_positives.iterrows():
            print(f"      [FP] {row['question'][:60]}... <- routed_from={row['expected_service']}")
            false_positive_examples.append({
                "question": row["question"],
                "choices": row["choices"],
                "routed_from": row["expected_service"],
            })

        return {
            "service": service,
            "total_examples": len(service_rows),

            "true_positive_count": len(true_positives),
            "false_negative_count": len(false_negatives),
            "false_positive_count": len(false_positives),
            
            "true_positive_examples": true_positive_examples,
            "false_negative_examples": false_negative_examples,
            "false_positive_examples": false_positive_examples,
        }


    def _save_profiles(self, profiles):
        path = self.results_dir / f"service_profiles_{len(self.services) - 1}_services.json"

        with open(path, "w") as f:
            json.dump(profiles, f, indent=2)

        print(f"[3] Service profiles saved to {path}")

        return path

    def build_profiles(self, df):
        print(f"[1] Building service profiles for {len(self.services) - 1} services")

        profiles = []
        for service in self.services:
            profile = self._build_service_profile(df, service)
            profiles.append(profile)

        print(f"[2] Built {len(profiles) - 1} service profiles")

        self._save_profiles(profiles)
        return profiles

