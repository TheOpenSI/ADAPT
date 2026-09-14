import json
import os
import sys
import ollama

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)

from ollama import chat

from data.dataset_builder.config import (
    GENERAL_SUBJECTS,
    SERVICE_SETS,
)


class ServiceDescriptionGenerator:

    def __init__(
        self,
        concept_path="concept_extraction.json",
        core_concept_path="core_concepts.json",
        model="qwen3:32b",
        verbose=True,
    ):
        self.concept_path = concept_path
        self.core_concept_path = core_concept_path
        self.model = model
        self.verbose = verbose

        self.client = ollama.Client(
            host="http://localhost:11435"
        )

        self.extracted_data = self._load_json(
            self.concept_path
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


    def generate_all_specialist_descriptions(self,
                                             output_path="specialist_descriptions.json"):

        specialists = SERVICE_SETS[10]

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

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        self._log(
            f"\n[DONE] Saved specialist descriptions to "
            f"{output_path}"
            )

        return results


    def _build_general_description_prompt(self, general_subject_concepts):

        sections = []

        for subject, concepts in general_subject_concepts.items():
            concepts_text = self._format_concepts(concepts)

            section = (
                f"SOURCE SUBJECT: {subject}\n"
                f"{concepts_text}"
            )

            sections.append(section)

        general_text = "\n\n".join(sections)

        prompt = f"""
You are constructing a semantic description for a General academic
routing service.

The General service represents several heterogeneous academic domains
rather than one single coherent subject.

GENERAL SOURCE SUBJECTS AND EXTRACTED CONCEPTS:

{general_text}

TASK:
Create one concise semantic description representing the knowledge
coverage of the General routing service.

For the description:
- Represent all five source subjects.
- Give approximately balanced semantic coverage to each source subject,
  regardless of how many extracted concepts are listed for that subject.
- Preserve the diversity of the General service rather than making it
  appear to represent one academic discipline.
- Mention each source domain once using its most representative knowledge
  areas.
- Summarize closely related concepts into broader but still meaningful
  academic areas.
- Every topic or knowledge area mentioned in the description must be
  directly supported by one or more supplied extracted concepts.
- You may paraphrase or group closely related supplied concepts, but do
  not add background knowledge from outside the supplied concepts.
- Do not allow a source subject with more extracted concepts to dominate
  the description.
- Do not list every individual concept.
- Prioritize distinct semantic coverage over elaboration.
- Avoid restating the same domain using different terminology.
- Describe General through its represented knowledge domains, not through
  its relationship to the specialist services.
- Do not define General using fallback or exclusion-based language such as
  "questions that do not match another service".
- Do not mention validation questions, concept extraction, routing
  experiments, or benchmarking.
- Do not use procedural wording such as "choose this service when".
- Use natural academic language.
- Prioritize domain-specific terminology over generic boilerplate.
- Keep the final description between 55 and 70 words.

Before returning the final description:
- Refine the description for clarity, coherence, and semantic precision.
- Check that all five source subjects are represented.
- Check that no single source subject dominates the description.
- Check that every substantive topic is supported by the supplied concepts.
- Check for semantic repetition: if a domain or topic has already been
  represented, do not describe it again using different wording.
- Remove unsupported additions, unnecessary repetition, and generic
  filler language.
- Confirm that the final description is between 55 and 70 words.

Return ONLY valid JSON in this format:

{{
    "subject": "general",
    "description": ""
}}
""".strip()

        return prompt

    def _generate_general_description(self):

        general_subject_concepts = {}

        for subject in GENERAL_SUBJECTS:

            general_subject_concepts[subject] = (self.extracted_data[subject]["extracted_concepts"])
        self._log("[GENERAL] Generating General description")
        prompt = self._build_general_description_prompt(general_subject_concepts)
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
        result["subject"] = "general"
        self._log(
            f"[GENERAL] "
            f"{len(result['description'].split())} words"
        )

        return result


    def build_all_service_descriptions(self, 
                                       specialist_path="specialist_descriptions.json", 
                                       output_path="all_service_descriptions.json"):

        specialist_descriptions = self._load_json(specialist_path)
        general_description = self._generate_general_description()

        all_services = specialist_descriptions.copy()
        all_services["general"] = general_description

        with open(output_path, "w") as f:
            json.dump(
                all_services,
                f,
                indent=2
            )

        self._log(
            f"[DONE] Saved {len(all_services)} services "
            f"to {output_path}"
        )

        return all_services


    def flatten_service_descriptions(
            self,
            path="all_service_descriptions.json"):

        with open(path, "r") as f:
            data = json.load(f)

        routing_descriptions = {
            subject: service_data["description"]
            for subject, service_data in data.items()
        }

        with open(path, "w") as f:
            json.dump(routing_descriptions, f, indent=2)

        self._log(f"[DONE] Flattened service descriptions in {path}")

        return routing_descriptions



if __name__ == "__main__":

    generator = ServiceDescriptionGenerator()

    # # 1. Regenerate all 10 specialist descriptions
    # generator.generate_all_specialist_descriptions(
    #     output_path="specialist_descriptions.json"
    # )

    # # 2. Generate General + combine with specialists
    # generator.build_all_service_descriptions(
    #     specialist_path="specialist_descriptions.json",
    #     output_path="all_service_descriptions.json"
    # )

    # must do otherwise services structure wont match
    generator.flatten_service_descriptions("all_service_descriptions.json")
    