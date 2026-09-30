import json
import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)
import ollama
import pandas as pd

from data.dataset.config import SERVICE_SETS

class ServiceConceptExtractor:

    def __init__(
        self,
        data_path="../service_desc_generator/validation_questions/all_validation_questions.json",
        model="qwen2.5:32b",
        verbose=True,
    ):
        """Initialize the extractor and load validation data."""

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
        """Load and validate the specialist-only validation dataset."""

        df = pd.read_json(self.data_path)
        self._log(f"[LOAD] validation data: {df.shape}")
        self._log(f"[LOAD] routing services: {df['expected_service'].nunique()}")
        self._log(f"[LOAD] original subjects: {df['subject'].nunique()}")


        assert "general" not in df["expected_service"].values, (
        "General queries must not be used for description construction."
        )

        expected_subjects = set(SERVICE_SETS[14])
        actual_subjects = set(df["subject"].unique())

        assert actual_subjects == expected_subjects, (
            f"Unexpected validation subjects. "
            f"Expected: {expected_subjects}, "
            f"got: {actual_subjects}"
        )

        return df


    def _group_by_subject(self):
        """
        Group validation examples by original MMLU subject.
        """
        groups = {
            subject: group.copy()
            for subject, group in self.df.groupby("subject")}

        self._log(f"[GROUP] Number of subject groups: {len(groups)}")

        for subject, group in groups.items():
            self._log(f"[GROUP] {subject}: {len(group)} questions")

        return groups


    def _format_questions(self, df):
        """
        Format one subject's questions and choices into a single text block for the LLM.
        Example:

        Question 1:
        Which of the following terms describes the body's ability to maintain its normal state?
        Choices:
        - Anabolism
        - Catabolism
        - Tolerance
        - Homeostasis
        """

        formatted_questions = []

        for i, row in df.reset_index(drop=True).iterrows(): 
            choices = "\n".join(
                f"- {choice}"
                for choice in row["choices"]
            )

            formatted_question = (
                f"Question {i + 1}:\n"
                f"{row['question']}\n"
                f"Choices:\n{choices}"
            )

            formatted_questions.append(formatted_question)

        return "\n\n".join(formatted_questions)


    def _build_concept_extraction_prompt(self,subject,formatted_questions):
        """
        Build the prompt used to extract concepts from one subject's validation questions.
        """

        prompt = f"""
You are analyzing validation questions from an academic subject.

SUBJECT:
{subject}

VALIDATION QUESTIONS:
{formatted_questions}

TASK:
Analyze all of the questions and answer choices above.

Extract the domain knowledge concepts that the questions are testing.

Use the answer choices only as context for understanding what knowledge
the question tests. Do NOT treat every answer choice or named entity as
a separate concept.

For extracted_concepts:
- Identify the specific domain knowledge concept, principle, structure,
  mechanism, theory, or process being tested by each question.
- Prefer meaningful and reusable domain concepts over isolated entities,
  answer options, or question-specific wording.
- Use the narrowest concept that accurately represents the knowledge
  being tested, while avoiding unnecessary details that apply only to
  one particular question.
- Merge concepts only when they are true semantic near-duplicates and
  represent essentially the same knowledge.
- Keep concepts separate when they represent genuinely different
  knowledge, even if they belong to the same broader topic.
- Ensure the extracted concept set collectively covers the full
  validation set.
- Multiple questions may map to the same concept when they genuinely
  test closely related knowledge.
- Do NOT force one concept per question.
- Exclude concepts that appear only as irrelevant distractor choices.
- Do not simply copy answer choices.
- Do not introduce concepts unsupported by the validation questions.
- Do not broaden specific concepts into vague umbrella categories.
- Use concise, natural-language noun phrases.
- Do not use snake_case.
- Do not answer the questions.
- There is no required number of extracted concepts.

Return ONLY valid JSON in this format:

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


    def _build_core_concept_prompt(self, subject, extracted_concepts):
        """Build the prompt for core-concept selection."""

        concepts_text = "\n".join(
            f"- {concept}"
            for concept in extracted_concepts
        )
        prompt = f"""
You are constructing a semantic profile for an academic routing service.

SUBJECT:
{subject}

EXTRACTED CONCEPTS:
{concepts_text}

TASK:
Select the core concepts that best characterize the knowledge domain
represented by this subject.

For core_concepts:
- Select 6-9 concepts when at least 6 valid concepts are available.
- Select only from the extracted concepts provided above.
- Do not introduce new concepts.
- Prefer concepts that are central, specific, and representative of the
  subject's knowledge domain.
- Prefer concepts that are specific and characteristic of this subject,
  rather than concepts that could broadly describe many academic domains.
- Avoid vague umbrella concepts.
- Avoid redundant or near-duplicate concepts.
- Select concepts from different important areas of the subject when possible.
  Avoid selecting several concepts that represent essentially the same subarea
  while excluding other clearly represented subareas.
- Do not broaden concepts beyond their original meaning.
- If fewer than 6 extracted concepts are available, retain all valid concepts.
- There is no need to select 9 concepts if fewer concepts adequately
  represent the domain.

Return ONLY valid JSON in this format:

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


    def extract_all_core_concepts(self, concept_path="concept_extraction.json", output_path="core_concepts.json"):
        """
        Select core concepts for all specialist subjects.
        """

        concept_data = self._load_extracted_concepts(concept_path)

        specialists = SERVICE_SETS[14]

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


if __name__ == "__main__":

    extractor = ServiceConceptExtractor()

    # Stage 1: validation questions -> extracted concepts
    extracted_concepts_results = extractor.extract_all(
        output_path="concept_extraction.json"
    )

    # Stage 2: extracted concepts -> specialist core concepts
    core_concepts_results = extractor.extract_all_core_concepts(
        concept_path="concept_extraction.json",
        output_path="core_concepts.json"
    )