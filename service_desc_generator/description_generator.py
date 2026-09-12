import os
import sys

sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/..")

import torch
import time

from sentence_transformers import CrossEncoder
from services.RouterBase import RouterBase


class CrossEncoderRouter(RouterBase):

    routing_method = "cross-encoder"

    def __init__(
        self,
        service_desc_path,
        model_name,
        confidence_mode="margin",
        confidence_threshold=None,
        verbose=False,
    ):

        super().__init__(service_desc_path, model_name)

        self.verbose = verbose
        self.confidence_mode = confidence_mode
        self.confidence_threshold = confidence_threshold

        print(
            f"[CrossEncoderRouter] Loading cross-encoder model '{model_name}'..."
        )

        load_start = time.perf_counter()

        self.model = CrossEncoder(
            model_name,
            device="cuda"
        )

        print(
            f"[CrossEncoderRouter] Model loaded in "
            f"{time.perf_counter() - load_start:.2f}s "
            f"— running on device: {self.model.device}"
        )

        print(
            f"[CrossEncoderRouter] Loaded {len(self.services)} services."
        )

        if self.verbose:

            print("  Services loaded:")

            for name, desc in self.services.items():

                preview = (
                    desc
                    if len(desc) <= 80
                    else desc[:77] + "..."
                )

                print(
                    f"    [{name}] "
                    f"({len(desc)} chars): "
                    f"{preview!r}"
                )

        print("[CrossEncoderRouter] Ready to route.")


    def route(self, query):

        # -------------------------
        # REMOVE GENERAL
        # -------------------------

        specialist_services = {}

        for name, desc in self.services.items():

            if name != "general":
                specialist_services[name] = desc


        if self.verbose:

            preview = (
                query
                if len(query) <= 80
                else query[:77] + "..."
            )

            print(
                f"  Scoring query ({len(query)} chars) "
                f"against {len(specialist_services)} specialists: "
                f"{preview!r}"
            )


        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()


        names = list(specialist_services.keys())

        pairs = []

        for service_desc in specialist_services.values():
            pairs.append(
                (query, service_desc)
            )


        scores = self.model.predict(pairs)


        if torch.cuda.is_available():
            torch.cuda.synchronize()


        latency_ms = (
            time.perf_counter() - start
        ) * 1000


        scores = [
            float(score)
            for score in scores
        ]


        # -------------------------
        # TOP 1
        # -------------------------

        top1_score = max(scores)

        top1_index = scores.index(
            top1_score
        )

        top1_service = names[
            top1_index
        ]


        # -------------------------
        # TOP 2
        # -------------------------

        scores_copy = scores.copy()

        scores_copy[
            top1_index
        ] = float("-inf")

        top2_score = max(
            scores_copy
        )

        top2_index = scores_copy.index(
            top2_score
        )

        top2_service = names[
            top2_index
        ]


        # -------------------------
        # MARGIN
        # -------------------------

        margin = (
            top1_score
            - top2_score
        )


        # -------------------------
        # MEAN
        # -------------------------

        mean_score = (
            sum(scores)
            / len(scores)
        )


        # -------------------------
        # STANDARD DEVIATION
        # -------------------------

        variance = 0

        for score in scores:

            variance += (
                score
                - mean_score
            ) ** 2


        variance = (
            variance
            / len(scores)
        )


        sd = (
            variance ** 0.5
        )


        # -------------------------
        # CHOOSE CONFIDENCE METHOD
        # -------------------------

        if self.confidence_mode == "maximum":

            confidence_score = (
                top1_score
            )


        elif self.confidence_mode == "margin":

            confidence_score = (
                margin
            )


        elif self.confidence_mode == "sd":

            confidence_score = (
                sd
            )


        else:

            raise ValueError(
                f"Unknown confidence mode: "
                f"{self.confidence_mode}"
            )


        # -------------------------
        # CONFIDENCE GATE
        # -------------------------

        if self.confidence_threshold is None:

            predicted_label = (
                top1_service
            )

        else:

            if (
                confidence_score
                >= self.confidence_threshold
            ):

                predicted_label = (
                    top1_service
                )

            else:

                predicted_label = (
                    "general"
                )


        # -------------------------
        # TOKEN COUNT
        # -------------------------

        prompt_tokens = 0

        for q, d in pairs:

            encoded = (
                self.model.tokenizer.encode(
                    q,
                    d,
                    truncation=True,
                    max_length=
                    self.model.max_seq_length,
                    add_special_tokens=True,
                )
            )

            prompt_tokens += len(
                encoded
            )


        # -------------------------
        # VERBOSE OUTPUT
        # -------------------------

        if self.verbose:

            print("  scores:")

            for i in range(
                len(names)
            ):

                marker = ""

                if (
                    names[i]
                    == top1_service
                ):
                    marker = " <-- top1"

                print(
                    f"    {names[i]}: "
                    f"{scores[i]:.4f}"
                    f"{marker}"
                )


            print(
                f"  Top 1: "
                f"{top1_service}: "
                f"{top1_score:.4f}"
            )

            print(
                f"  Top 2: "
                f"{top2_service}: "
                f"{top2_score:.4f}"
            )

            print(
                f"  Margin: "
                f"{margin:.4f}"
            )

            print(
                f"  SD: "
                f"{sd:.4f}"
            )

            print(
                f"  Confidence mode: "
                f"{self.confidence_mode}"
            )

            print(
                f"  Confidence score: "
                f"{confidence_score:.4f}"
            )

            print(
                f"  Threshold: "
                f"{self.confidence_threshold}"
            )

            print(
                f"  -> {predicted_label} "
                f"| {latency_ms:.1f}ms "
                f"| {prompt_tokens} tok"
            )


        # -------------------------
        # RETURN
        # -------------------------

        return {

            "routing_method":
                self.routing_method,

            "query":
                query,

            "raw_response":
                None,

            "predicted_label":
                predicted_label,

            "model_name":
                self.model_name,

            "latency_ms":
                latency_ms,

            "prompt_tokens":
                prompt_tokens,

            "completion_tokens":
                0,

            "total_tokens":
                prompt_tokens,

            "scores":
                dict(
                    zip(
                        names,
                        scores
                    )
                ),

            "top1_service":
                top1_service,

            "top1_score":
                top1_score,

            "top2_service":
                top2_service,

            "top2_score":
                top2_score,

            "margin":
                margin,

            "mean_score":
                mean_score,

            "sd":
                sd,

            "confidence_mode":
                self.confidence_mode,

            "confidence_score":
                confidence_score,

            "confidence_threshold":
                self.confidence_threshold,
        }