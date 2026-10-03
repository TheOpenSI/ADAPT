import json
import os
import sys

project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

sys.path.insert(0, project_root)

import ollama

from data.ScienceQA.specialists_dataset.scienceqa_config import SCIENCEQA_SERVICE_SETS


class ServiceDescriptionGenerator:

    def __init__(
        self,
        service_count,
        general_concept_path,
        core_concept_path,
        output_path,
        model,
        verbose=True,
    ):
        self.service_count = service_count
        self.general_concept_path = general_concept_path
        self.core_concept_path = core_concept_path
        self.output_path = output_path

        self.model = model
        self.verbose = verbose

        self.client = ollama.Client(
            host="http://localhost:11435"
        )

        self.extracted_data = self._load_json(
            self.general_concept_path
        )

        self.core_data = self._load_json(
            self.core_concept_path
        )

    def _log(self, message):
        if self.verbose:
            print(message)


    def _load_json(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        self._log(
            f"[LOAD] {path}: {len(data)} subjects"
        )

        return data

    def _format_concepts(self, concepts):
        return "\n".join(
            f"- {concept}"
            for concept in concepts
        )


    def _build_specialist_description_prompt(
            self,
            subject,
            extracted_concepts,
            core_concepts
        ):

        core_text = self._format_concepts(core_concepts)
        extracted_text = self._format_concepts(extracted_concepts)

        prompt = f"""
You are constructing a semantic description for an academic specialist
routing service.

SUBJECT:
{subject}

CORE CONCEPTS:
{core_text}

FULL EXTRACTED CONCEPTS:
{extracted_text}

TASK:
Create one concise semantic description of the academic knowledge covered
by this specialist service.

The core concepts represent the strongest anchors of the subject.
The full extracted concepts provide broader coverage of the knowledge
represented by the subject.

For the description:
- Use the core concepts as the primary anchors.
- Use the full extracted concepts to preserve broader subject coverage.
- Represent multiple important areas of the subject when supported by
  the concepts.
- Summarize related concepts into meaningful domain areas rather than
  listing every concept individually.
- Begin directly with the subject name or subject-specific knowledge.
- Avoid generic boilerplate openings such as "This domain encompasses",
  "This service covers", or "The academic knowledge domain encompasses".
- Prioritize domain-specific terminology over generic descriptive language.
- Describe the academic knowledge domain rather than individual questions.
- Use only information supported by the supplied concepts.
- Do not introduce unsupported topics.
- Do not mention validation questions, concept extraction, routing
  experiments, or benchmarking.
- Do not use procedural wording such as "choose this service when".
- Use concise, natural academic language.
- Write between 40 and 60 words.

Before returning the final description:
- Refine the description for clarity, coherence, and semantic precision.
- Remove unnecessary repetition and generic boilerplate language.
- Ensure the wording is natural and academically appropriate.
- Ensure the subject name is written naturally rather than in snake_case.
- Check that the description represents both the core concepts and the
  broader extracted concepts.
- Check that no unsupported concepts have been introduced.

Return ONLY valid JSON in this format:

{{
    "subject": "{subject}",
    "description": ""
}}
""".strip()

        return prompt

    def _generate_specialist_description(self, subject):
        extracted_concepts = (self.extracted_data[subject]["extracted_concepts"])
        core_concepts = (self.core_data[subject]["core_concepts"])
        self._log(f"[SPECIALIST] Generating description: {subject}")

        prompt = self._build_specialist_description_prompt(
            subject,
            extracted_concepts,
            core_concepts
        )

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={"temperature": 0}
        )
        result = json.loads(response.message.content)
        result["subject"] = subject

        self._log(
            f"[SPECIALIST] {subject}: "
            f"{len(result['description'].split())} words")

        return result


    def generate_all_specialist_descriptions(self):

        specialists = SCIENCEQA_SERVICE_SETS[self.service_count]

        results = {}
        total = len(specialists)

        for i, subject in enumerate(specialists,start=1):
            self._log(f"\n[{i}/{total}] "
                      f"Generating specialist description: {subject}")

            try:
                result = self._generate_specialist_description(subject)
                results[subject] = result

            except Exception as e:
                self._log(f"[ERROR] {subject}: {e}")
                results[subject] = {"subject": subject,"error": str(e)}

            with open(self.output_path, "w") as f:
                json.dump(results, f, indent=2)

        self._log(
            f"\n[DONE] Saved specialist descriptions to "
            f"{self.output_path}"
            )

        return results



    def flatten_service_descriptions(
            self):

        with open(self.output_path, "r") as f:
            data = json.load(f)

        routing_descriptions = {
            subject: service_data["description"]
            for subject, service_data in data.items()
        }

        with open(self.output_path, "w") as f:
            json.dump(routing_descriptions, f, indent=2)

        self._log(f"[DONE] Flattened service descriptions in {self.output_path}")

        return routing_descriptions



if __name__ == "__main__":
    generator = ServiceDescriptionGenerator(
        service_count=8,
        model="qwen3:32b",

        general_concept_path=(
            "data/ScienceQA/descriptions/"
            "general_concepts_extraction.json"
        ),
        core_concept_path=(
            "data/ScienceQA/descriptions/"
            "core_concepts.json"
        ),
        output_path=(
            "data/ScienceQA/descriptions/"
            "service_descriptions_8.json"
        ),
    )
    # generator.generate_all_specialist_descriptions()


    # must do otherwise services structure wont match
    generator.flatten_service_descriptions()
    


    