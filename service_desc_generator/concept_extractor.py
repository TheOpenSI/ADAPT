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
import pandas as pd

# from data.MedMCQA.specialists_dataset.medmcqa_config import MEDMCQA_SERVICE_SETS
from data.MASSIVE.specialists.massive_config import (
    MASSIVE_SERVICE_SETS
)


class ServiceConceptExtractor:

    def __init__(
        self,
        data_path,
        service_set,
        model="qwen2.5:32b",
        verbose=True,
    ):
        """Initialize the extractor and load validation data."""

        self.service_set = service_set

        self.data_path = data_path
        self.model = model
        self.verbose = verbose
        self.client = ollama.Client(host="http://localhost:11435")

        self.df = self._load_data()



    def _log(self, message):
        """Print messages when verbose mode is enabled."""

        if self.verbose:
            print(message)


    def _load_data(self):
        """Load the specialist-only validation dataset."""

        df = pd.read_json(self.data_path)
        self._log(f"[LOAD] validation data: {df.shape}")
        self._log(f"[LOAD] routing services: {df['expected_service'].nunique()}")
        self._log(
            f"[LOAD] specialist subjects: "
            f"{df['subject'].nunique()}"
        )
        actual_subjects = set(df["subject"].unique())
        expected_subjects = set(self.service_set)
        
        assert actual_subjects == expected_subjects, (
            f"Expected subjects: {expected_subjects}, "
            f"got: {actual_subjects}"
        )

        return df


    def _group_by_subject(self):
        """
        Group validation examples by specialist subject.
        """
        groups = {
            subject: group.copy()
            for subject, group in self.df.groupby("subject")}

        self._log(f"[GROUP] Number of subject groups: {len(groups)}")

        for subject, group in groups.items():
            self._log(f"[GROUP] {subject}: {len(group)} questions")

        return groups


    # def _format_questions(self, df):
    #     """
    #     Format one subject's questions and choices into a single text block for the LLM.
    #     Example:

    #     Question 1:
    #     Which of the following terms describes the body's ability to maintain its normal state?
    #     Choices:
    #     - Anabolism
    #     - Catabolism
    #     - Tolerance
    #     - Homeostasis
    #     """

    #     formatted_questions = []

    #     for i, row in df.reset_index(drop=True).iterrows(): 
    #         choices = "\n".join(
    #             f"- {choice}"
    #             for choice in row["choices"]
    #         )

    #         formatted_question = (
    #             f"Question {i + 1}:\n"
    #             f"{row['question']}\n"
    #             f"Choices:\n{choices}"
    #         )

    #         formatted_questions.append(formatted_question)

    #     return "\n\n".join(formatted_questions)


    def _format_questions(self, df):
        """Format utterances for concept extraction."""

        formatted_questions = []

        for i, row in df.reset_index(drop=True).iterrows():

            formatted_question = (
                f"Utterance {i + 1}:\n"
                f"{row['question']}"
            )

            formatted_questions.append(
                formatted_question
            )

        return "\n\n".join(
            formatted_questions
        )


#     def _build_concept_extraction_prompt(self,subject,formatted_questions):
#         """
#         Build the prompt used to extract concepts from one validation subject.
#         """

#         prompt = f"""
# You are analyzing validation questions from a specialist academic domain.

# SUBJECT:
# {subject}

# VALIDATION QUESTIONS:
# {formatted_questions}

# TASK:
# Analyze all of the questions and answer choices above.

# Extract the domain knowledge concepts that the questions are testing.

# Use the answer choices only as context for understanding what knowledge
# the question tests. Do NOT treat every answer choice or named entity as
# a separate concept.

# For extracted_concepts:
# - Identify the specific domain knowledge concept, principle, structure,
#   mechanism, theory, or process being tested by each question.
# - Prefer meaningful and reusable domain concepts over isolated entities,
#   answer options, or question-specific wording.
# - Use the narrowest concept that accurately represents the knowledge
#   being tested, while avoiding unnecessary details that apply only to
#   one particular question.
# - Merge concepts only when they are true semantic near-duplicates and
#   represent essentially the same knowledge.
# - Keep concepts separate when they represent genuinely different
#   knowledge, even if they belong to the same broader topic.
# - Ensure the extracted concept set collectively covers the full
#   validation set.
# - Multiple questions may map to the same concept when they genuinely
#   test closely related knowledge.
# - Do NOT force one concept per question.
# - Exclude concepts that appear only as irrelevant distractor choices.
# - Do not simply copy answer choices.
# - Do not introduce concepts unsupported by the validation questions.
# - Do not broaden specific concepts into vague umbrella categories.
# - Use concise, natural-language noun phrases.
# - Do not use snake_case.
# - Do not answer the questions.
# - There is no required number of extracted concepts.

# Return ONLY valid JSON in this format:

# {{
#     "subject": "{subject}",
#     "extracted_concepts": []
# }}
# """.strip()

#         return prompt
    
    def _build_concept_extraction_prompt(
        self,
        subject,
        formatted_questions
    ):
        """
        Build the prompt used to broadly extract functional
        concepts from one MASSIVE specialist subject.
        """

        prompt = f"""
You are analyzing validation questions associated with a specialist
service.

SUBJECT:
{subject}

VALIDATION QUESTIONS:
{formatted_questions}

TASK:
Analyze all validation questions and broadly extract the functional
concepts represented by this specialist service.

This stage prioritizes coverage. Capture the meaningful user goals,
requested actions, supported capabilities, operational functions, and
information needs represented across the validation set.

For extracted_concepts:

- Identify the functional concept represented by each important type of
  validation question.
- Base every concept on evidence from the supplied validation questions.
- Prefer meaningful and reusable concepts over question-specific wording.
- Use concepts that describe what users expect the service to do.
- Ensure that the extracted concepts collectively cover the important
  request patterns represented across the validation set.
- Preserve fine-grained concepts when they may represent different request
  patterns, service operations, or capability variations.
- Merge only obvious duplicates that express essentially the same concept
  using identical or nearly identical wording.
- Multiple validation questions may map to the same concept.
- Do not force one concept per validation question.
- Do not extract academic knowledge areas, theories, or general subject
  names.
- Do not introduce concepts unsupported by the validation questions.
- Do not copy complete validation questions.
- Do not extract specific people, organizations, locations, dates, times,
  products, restaurants, songs, or other example-specific named entities.
- Treat named entities as context for identifying the underlying user goal
  or service capability.
- Ignore isolated, incomplete, malformed, or apparently mislabeled
  validation questions when they conflict with the dominant functional
  scope of the service.
- Avoid vague concepts such as "assistance", "information",
  "communication", "technology", or "user requests".
- Use concise, lowercase, natural-language phrases.
- Use spaces between words.
- Do not use snake_case, camelCase, service-label syntax, or underscores.
- Do not answer or execute the validation questions.
- There is no required number of extracted concepts. Include all meaningful
  functional concepts needed to cover the validation set.
- Do not aggressively consolidate related concepts in this stage. A later
  stage will merge related and parameter-specific concepts into a compact
  core capability profile.

Before returning the result, verify that:

- every concept is supported by the supplied validation questions;
- every concept describes a user goal, requested action, supported
  capability, operational function, or information need;
- no concept contains an underscore;
- no concept is a complete validation question;
- no concept contains an example-specific named entity;
- obvious exact or near-exact duplicates have been removed;
- the concept set provides broad coverage of the validation questions.

Return ONLY valid JSON in exactly this structure:

{{
    "subject": "{subject}",
    "extracted_concepts": []
}}
""".strip()

        return prompt

    


    def _call_llm(self, subject, df):
        """Extract concepts for one specialist subject."""

        self._log(f"[EXTRACT] Processing {subject}: {len(df)} questions")

        formatted_questions = self._format_questions(df)
        prompt = self._build_concept_extraction_prompt(subject,formatted_questions)

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        result = json.loads(response.message.content)

        self._log(
            f"[EXTRACT] {subject}: "
            f"{len(result['extracted_concepts'])} concepts"
        )

        return result


    def extract_all(self, output_path="concept_extraction.json"):
        """Extract and save concepts for all specialists."""

        groups = self._group_by_subject()
        results = {}
        total = len(groups)

        for i, (subject, df) in enumerate(groups.items(),start=1):
            self._log(
                f"\n[{i}/{total}] Extracting {subject} "
                f"({len(df)} questions)"
            )
            try:
                result = self._call_llm(subject,df)
                results[subject] = result

            except Exception as e:
                self._log(f"[ERROR] {subject}: {e}")

                results[subject] = {
                    "subject": subject,
                    "error": str(e)
                }

            with open(output_path, "w") as f:
                json.dump(results,f,indent=2)

        self._log(f"\n[DONE] Saved results to {output_path}")

        return results


#     def _build_core_concept_prompt(self, subject, extracted_concepts):
#         """Build the prompt for core-concept selection."""

#         concepts_text = "\n".join(
#             f"- {concept}"
#             for concept in extracted_concepts
#         )
#         prompt = f"""
# You are constructing a semantic profile for an academic routing service.

# SUBJECT:
# {subject}

# EXTRACTED CONCEPTS:
# {concepts_text}

# TASK:
# Select the core concepts that best characterize the knowledge domain
# represented by this subject.

# For core_concepts:
# - Select 6-9 concepts when at least 6 valid concepts are available.
# - Select only from the extracted concepts provided above.
# - Do not introduce new concepts.
# - Prefer concepts that are central, specific, and representative of the
#   subject's knowledge domain.
# - Prefer concepts that are specific and characteristic of this subject,
#   rather than concepts that could broadly describe many academic domains.
# - Avoid vague umbrella concepts.
# - Avoid redundant or near-duplicate concepts.
# - Select concepts from different important areas of the subject when possible.
#   Avoid selecting several concepts that represent essentially the same subarea
#   while excluding other clearly represented subareas.
# - Do not broaden concepts beyond their original meaning.
# - If fewer than 6 extracted concepts are available, retain all valid concepts.
# - There is no need to select 9 concepts if fewer concepts adequately
#   represent the domain.

# Return ONLY valid JSON in this format:

# {{
#     "subject": "{subject}",
#     "core_concepts": []
# }}
# """.strip()

#         return prompt


    def _build_core_concept_prompt(
        self,
        subject,
        extracted_concepts
):
        """
        Build the prompt used to consolidate broad extracted
        concepts and produce the core functional concepts for
        one MASSIVE specialist service.
        """

        concepts_text = "\n".join(
            f"- {concept}"
            for concept in extracted_concepts
    )

        prompt = f"""
You are constructing a compact semantic capability profile for a
specialist routing service.

SUBJECT:
{subject}

BROADLY EXTRACTED CONCEPTS:
{concepts_text}

TASK:
Consolidate, normalize, and select the core functional concepts that best
characterize this specialist service.

The extracted concepts were generated broadly to maximize coverage. They
may contain duplicates, semantic near-duplicates, different wording for
the same operation, parameter-specific variants, content-specific
variants, overly specific concepts, and concepts derived from noisy
validation questions.

Do not merely select items from the supplied list. Merge and rewrite
related concepts into a concise set of semantically distinct core
capabilities suitable for constructing an IR service description.

For core_concepts:

- Identify the underlying functional operation represented by each group
  of related extracted concepts.
- Merge concepts that describe substantially the same capability using
  different wording.
- Merge concepts that differ only by parameter values, content topics,
  target entities, contexts, or example-specific details.
- Distinguish the underlying service operation from its arguments and
  parameters.
- Treat dates, times, recurrence frequencies, labels, locations,
  participants, recipients, rooms, devices, message contents, news topics,
  food types, event types, and similar details as parameters rather than
  separate capabilities.
- Rewrite merged concepts as concise, lowercase, natural-language noun
  phrases.
- Use spaces between words.
- Do not use snake_case, camelCase, service-label syntax, or underscores.
- Keep concepts separate only when they represent meaningfully different
  user goals, requested actions, information needs, operational functions,
  or service capabilities.
- Prefer capabilities that are central to the service and supported by
  multiple extracted concepts or recurring validation patterns.
- Preserve enough semantic diversity to match the major kinds of requests
  handled by the service.
- Prefer service-specific concepts that help distinguish this specialist
  from other routing services.
- Avoid vague umbrella concepts that may apply to many different services.
- Exclude concepts based primarily on isolated, malformed, noisy, or
  apparently mislabeled validation questions.
- Do not include specific people, organizations, locations, dates, times,
  products, restaurants, songs, or other example-specific entities.
- Select 6-9 core concepts when at least 6 distinct capabilities are
  genuinely supported.
- If fewer than 6 distinct capabilities are supported, return only the
  valid supported concepts.
- Do not create additional concepts merely to reach a target number.
- Every core concept must be grounded in one or more extracted concepts,
  but it does not need to repeat their exact wording.

Before returning the result, verify that:

- every core concept is grounded in the broadly extracted concepts;
- every core concept describes a concrete functional capability, user
  goal, requested action, operational function, or information need;
- no core concept contains an underscore;
- no core concept represents only a parameter value, message topic,
  content category, target entity, location, recurrence frequency, or
  example-specific detail;
- no two core concepts differ only by wording, parameters, content, or
  target entities;
- no two core concepts describe substantially the same capability;
- the final concept set covers the principal functional scope of the
  specialist service;
- the final concepts are specific enough to distinguish this service from
  other specialist services.

Return ONLY valid JSON in exactly this structure:

{{
    "subject": "{subject}",
    "core_concepts": []
}}
""".strip()

        return prompt

    def _load_extracted_concepts(self,path="concept_extraction.json"):
        """Load extracted concepts from Stage 1."""

        with open(path, "r") as f:
            data = json.load(f)

        self._log(f"[LOAD] Extracted concepts for {len(data)} subjects")

        return data


    def _extract_core_concepts(self, subject, extracted_concepts):
        """Select core concepts for one specialist subject."""

        self._log(
            f"[CORE] Processing {subject}: "
            f"{len(extracted_concepts)} extracted concepts"
            )

        prompt = self._build_core_concept_prompt(subject, extracted_concepts)

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0
            }
    )

        result = json.loads(response.message.content)

        self._log(
            f"[CORE] {subject}: "
            f"{len(result['core_concepts'])} core concepts"
        )

        return result


    def extract_all_core_concepts(self, general_concept_path, output_path):
        """
        Select core concepts for all specialist subjects.
        """

        concept_data = self._load_extracted_concepts(general_concept_path)

        specialists = self.service_set

        results = {}
        total = len(specialists)

        for i, subject in enumerate(specialists, start=1):
            self._log(f"\n[{i}/{total}] Selecting core concepts: {subject}")
            try:

                extracted_concepts = (concept_data[subject]["extracted_concepts"])
           
                result = self._extract_core_concepts(
                    subject,
                    extracted_concepts
                )
                results[subject] = result

            except Exception as e:
                self._log(f"[ERROR] {subject}: {e}")

                results[subject] = {
                    "subject": subject,
                    "error": str(e)
                }

            with open(output_path, "w") as f:
                json.dump(results,f,indent=2)

        self._log(f"\n[DONE] Saved core concepts to {output_path}")

        return results


# if __name__ == "__main__":

#     SERVICE_ID = 1

#     output_dir = os.path.join(
#         project_root,
#         "data",
#         "MedMCQA",
#         "descriptions"
#     )

#     os.makedirs(
#         output_dir,
#         exist_ok=True
#     )

#     validation_path = os.path.join(
#         project_root,
#         "data",
#         "MedMCQA",
#         "validation_questions",
#         f"medmcqa_validation_questions_{SERVICE_ID}.json"
#     )

#     print(f"[PATH] Validation data: {validation_path}")

#     if not os.path.exists(validation_path):
#         raise FileNotFoundError(
#             f"Validation data not found: {validation_path}"
#         )

#     extractor = ServiceConceptExtractor(
#         data_path=validation_path,
#         service_set=MEDMCQA_SERVICE_SETS[SERVICE_ID],
#         model="qwen2.5:32b"
#     )

#     general_concept_path = os.path.join(
#         output_dir,
#         "general_concepts_extraction_1_subject.json"
#     )

#     core_concept_path = os.path.join(
#         output_dir,
#         "core_concepts_1_subject.json"
#     )

#     # Stage 1:
#     # validation questions -> extracted concepts
#     extractor.extract_all(
#         output_path=general_concept_path
#     )

#     # Stage 2:
#     # extracted concepts -> specialist core concepts
#     extractor.extract_all_core_concepts(
#         general_concept_path=general_concept_path,
#         output_path=core_concept_path
#     )


if __name__ == "__main__":

    SERVICE_ID = 9

    output_dir = os.path.join(
        project_root,
        "data",
        "MASSIVE",
        "descriptions"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    validation_path = os.path.join(
        project_root,
        "data",
        "MASSIVE",
        "validation_questions",
        f"massive_validation_questions_{SERVICE_ID}.json"
    )

    print(f"[PATH] Validation data: {validation_path}")

    if not os.path.exists(validation_path):
        raise FileNotFoundError(
            f"Validation data not found: {validation_path}"
        )

    extractor = ServiceConceptExtractor(
        data_path=validation_path,
        service_set=MASSIVE_SERVICE_SETS[SERVICE_ID],
        model="qwen2.5:32b"
    )

    extracted_concept_path = os.path.join(
        output_dir,
        f"extracted_concepts_{SERVICE_ID}_subjects.json"
    )

    core_concept_path = os.path.join(
        output_dir,
        f"core_concepts_{SERVICE_ID}_subjects.json"
    )

    # Stage 1:
    # validation utterances -> extracted functional concepts
    extractor.extract_all(
        output_path=extracted_concept_path
    )

    # Stage 2:
    # extracted functional concepts -> core service concepts
    extractor.extract_all_core_concepts(
        general_concept_path=extracted_concept_path,
        output_path=core_concept_path
    )