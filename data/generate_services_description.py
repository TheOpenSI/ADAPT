import os
import sys
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

from data.service_description_builder.service_description_builder import ServiceDescriptionGenerator
from data.service_description_builder.service_profile_builder import ServiceProfileBuilder 
from data.dataset_builder.config import SERVICE_SETS
from data.service_description_builder.profiling_dataset_builder import ProfilingDatasetBuilder
from data.service_description_builder.routing_tester import RoutingTester


from pathlib import Path
import json

PROFILE_PATH = Path("../data/service_profiles_10_services.json")

def main():

    subjects = SERVICE_SETS[10]
    services = subjects + ["general"]

    # Check if profiles already exist
    if PROFILE_PATH.exists():

        print(
            f"[FOUND] Existing profiles: {PROFILE_PATH}"
        )

        print(
            "[SKIP] Dataset, routing, and profile generation"
        )

        with open(PROFILE_PATH, "r") as f:
            profiles = json.load(f)

    else:

        print(
            f"[NOT FOUND] {PROFILE_PATH}"
        )

        print(
            "[RUN] Building profiling pipeline..."
        )

        # 1. Build dataset
        dataset_builder = ProfilingDatasetBuilder(
            subjects=subjects
        )

        df = dataset_builder.build_dataset()

        # 2. Route
        routing_profiler = RoutingTester(
            services=services
        )

        routing_results = routing_profiler.run_test(
            df
        )

        # 3. Build profiles
        profile_builder = ServiceProfileBuilder(
            services=services
        )

        profiles = profile_builder.build_profiles(
            routing_results
        )

    # 4. Always generate descriptions
    description_generator = ServiceDescriptionGenerator(
        services=services
    )

    descriptions = (
        description_generator
        .generate_descriptions(profiles)
    )

if __name__ == "__main__":
    main()