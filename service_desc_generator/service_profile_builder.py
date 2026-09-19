import json
import os
import sys

sys.path.append(
    f"{os.path.dirname(os.path.abspath(__file__))}/.."
)

import ollama
import pandas as pd


class ServiceProfileGenerator:
    def __init__(
        self,
        data_path="../service_desc_generator/dataset/validation_service_all.json",
        model="qwen3:32b",
        verbose=True,
    ):
        self.data_path = data_path
        self.model = model
        self.verbose = verbose

        self.client = ollama.Client(host="http://localhost:11435")

        self.df = self._load_data()

    def _log(self, message):

        if self.verbose:
            print(message)

    def _load_data(self):

        df = pd.read_json(self.data_path)

        self._log(
            f"[LOAD] validation data: {df.shape}"
        )

        self._log(
            f"[LOAD] routing services: "
            f"{df['expected_service'].nunique()}"
        )

        self._log(
            f"[LOAD] subjects: "
            f"{sorted(df['subject'].unique())}"
        )

        return df

    def _group_by_subject(self):
        """
        Group validation examples by original MMLU subject.
        """

        groups = {
            subject: group.copy()
            for subject, group in self.df.groupby("subject")
        }

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

    

    def _build_profile_prompt(
        self,
        subject,
        formatted_questions
    ):

        """
        Build the prompt to create specialists descriptions.  
        """

        prompt = f"""
You are analyzing validation questions from an academic specialist.

SUBJECT:
{subject}

VALIDATION QUESTIONS:
{formatted_questions}

TASK

Analyze all validation questions and infer the specialist represented by them.

The goal is to create a specialist profile for routing future user requests.

function:
- 15-35 words
- one sentence only
- describe the specialist's core expertise
- use domain terminology
- focus on broad capabilities, not individual facts
- summarize what the specialist can reason about or solve
- do not mention specific validation questions

tags:
- 6-20 tags
- lowercase
- use hyphen-separated format
- prefer specific technical concepts over broad disciplines
- include theories, laws, structures, methods, doctrines, named concepts, and specialized terminology
- maximize specialist distinctiveness
- maximize routing utility
- only include concepts directly supported by the questions
- concepts may be included even if they appear only once when they strongly characterize the specialist
- every tag must be traceable to one or more validation questions
- do not invent theories, laws, doctrines, or concepts that are not explicitly mentioned or clearly tested
- avoid generic labels when a more specific concept exists
- avoid duplicates and near-duplicates

Do not include:
- isolated facts that do not represent a meaningful area of expertise
- tags that overlap heavily

Return JSON only:

{{
  "function": "",
  "tags": []
}}
""".strip()


        return prompt


    def _generate_profile(
        self,
        subject,
        df
    ):
        """
        Generate a specialist description profile for one subject.
        """

        self._log(f"[EXTRACT] Processing {subject}: {len(df)} questions")

        formatted_questions = self._format_questions(df)
        prompt = self._build_profile_prompt(subject,formatted_questions)

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

        self._log(
            f"[PROFILE] {subject}: "
            f"{len(result['tags'])} tags"
        )

        assert "function" in result
        assert "tags" in result

        return result

    def generate_all(
        self,
        output_path="service_profiles.json"
    ):

        groups = self._group_by_subject()
        results = {}
        total = len(groups)

        for i, (subject, df) in enumerate(groups.items(), start=1):
            self._log(
                f"\n[{i}/{total}] {subject}"
            )

            try:
                result = (
                    self._generate_profile(subject, df)
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

        self._log(f"\n[DONE] Saved results to {output_path}")

        return results


if __name__ == "__main__":

    generator = ServiceProfileGenerator()

    generator.generate_all(
        output_path="../service_desc_generator/dataset/service_profiles.json"
    )

