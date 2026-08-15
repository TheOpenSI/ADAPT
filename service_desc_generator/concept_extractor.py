import json
import pandas as pd
from ollama import chat

class ServiceConceptExtractor:

    def __init__(
        self,
        data_path="../data/dataset/validation_service_all.json",
        model="qwen2.5:14b",
        verbose=True,
    ):
        self.data_path = data_path
        self.model = model
        self.verbose = verbose
        self.df = self._load_data()


    def _log(self, message):
        if self.verbose:
            print(message)

    def _load_data(self):
        df = pd.read_json(self.data_path)
        self._log(f"[LOAD] validation data: {df.shape}")
        self._log(f"[LOAD] routing services: {df['expected_service'].nunique()}")
        return df

    def _group_by_service(self):
        """
        Group validation rows by expected service.

        Specialist services contain their service DataFrame.
        General is further grouped by original subject.
        """

        self._log("\n[GROUP] grouping validation data by subject...")

        flat_groups = {}
        for expected_service, service_df in self.df.groupby("expected_service"):
            if expected_service == "general":
                for subject, subject_df in service_df.groupby("subject"):
                    flat_groups[subject] = subject_df.copy()
            else:
                flat_groups[expected_service] = service_df.copy()

        self._log(f"[GROUP] {len(flat_groups)} subjects to process")
        return flat_groups


    def _format_questions(self, df):
        blocks = []
        for i, row in df.reset_index(drop=True).iterrows():
            choices = "\n".join(f"- {c}" for c in row["choices"])
            blocks.append(
                f"QUESTION {i + 1}:\n{row['question']}\n\nCHOICES:\n{choices}"
            )
        return "\n\n".join(blocks)


    def _build_concept_extraction_prompt(
            self,
            service,
            formatted_questions
            ):
        """
        Build the prompt used to extract concepts
        from a service's validation questions.
        """

        prompt = f"""
You are constructing a semantic service profile for a routing benchmark.

At routing time, the service is represented by this description.
Therefore, the profile must contain concrete and discriminative domain
terminology that helps distinguish this service from other services.

SERVICE:
{service}

VALIDATION QUESTIONS:
{formatted_questions}

TASK:
Analyze all of the questions and answer choices above.

Extract the domain concepts that the questions are testing.

Use the answer choices only as context for understanding the question.
Do NOT treat every answer choice or named entity as a separate concept.

For extracted_concepts:
- Identify the specific knowledge concept being tested by each question.
- Prefer meaningful domain concepts over isolated entities or answer options.
- Merge closely related concepts.
- Exclude concepts that appear only as irrelevant distractor choices.
- Do not simply copy terms from the answer choices.
- Ensure that concepts represented across the full set of questions are considered.

For core_concepts:
- Select 6-9 of the most discriminative concepts from extracted_concepts
  when at least 6 valid concepts are available.
- If fewer than 6 valid extracted_concepts are available, retain all of
  them. Never invent, broaden, or duplicate concepts merely to reach a
  target count.
- If extracted_concepts has more than 9 items, select the 6-9 MOST
  discriminative and specific ones. Do not average or merge distinct
  concepts into broader categories simply to cover more ground.
  Dropping less-central concepts is preferred over abstracting them into
  a vaguer term.
- If extracted_concepts has 6-9 items, you may keep most or all of them,
  while consolidating true near-duplicates.
- A core concept must be specific enough that it would NOT equally
  describe a different, closely related subject. For example,
  "neuroscience" is too broad, while "hypothalamic function" is
  appropriately specific.
- Select concepts from across the full extracted_concepts list rather
  than simply choosing concepts that appear earliest.

For scope:
- Write one concise statement, maximum 40 words, describing the semantic
  domain represented by the questions.
- Ground the scope in the core concepts rather than merely repeating
  the service name.

Do not answer the questions.
Do not introduce concepts unsupported by the questions.

Return ONLY valid JSON in this format:
{{
    "service": "{service}",
    "extracted_concepts": [],
    "core_concepts": [],
    "scope": ""
}}

""".strip()

        return prompt


    def _call_llm(self, prompt):
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        return json.loads(response.message.content)


    def _extract_one(self, subject, df):
        formatted = self._format_questions(df)
        prompt = self._build_concept_extraction_prompt(subject, formatted)
        return self._call_llm(prompt)


    def run_pipeline(self, out_path="../service_desc_generator/concept_extraction.json"):
        grouped = self._group_by_service()
        results = {}
        total = len(grouped)

        for idx, (subject, df) in enumerate(grouped.items(), start=1):
            self._log(f"\n[{idx}/{total}] extracting: {subject} ({len(df)} questions)")
            try:
                results[subject] = self._extract_one(subject, df)
            except Exception as e:
                self._log(f"[{idx}/{total}] FAILED: {subject} — {e}")
                results[subject] = {"error": str(e)}

            # save after every subject so a crash doesn't lose prior work
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        ok = sum(1 for v in results.values() if "error" not in v)
        self._log(f"\n[DONE] {ok}/{total} succeeded. Saved to {out_path}")
        return results


if __name__ == "__main__":
    extractor = ServiceConceptExtractor()
    extractor.run_pipeline()


#     def _get_specialist_services(self):
#         """
#         Return all services except general.
#         """
        
#         services = (
#             self.df[
#                 self.df["expected_service"] != "general"
#             ]["expected_service"]
#             .unique()
#             .tolist()
#         )

#         return services

#     def _get_service_questions(self, service):
#         """
#         Get all validation rows belonging to
#         one specialist service.
#         """

#         service_df = self.df[
#             self.df["expected_service"] == service
#         ].copy()

#         return service_df

#     def _get_general_subjects(self):
#         """
#         Separate the original subjects that
#         make up the general service.
#         """

#         general_df = self.df[
#             self.df["expected_service"] == "general"
#         ]

#         subjects = (
#             general_df["subject"]
#             .unique()
#             .tolist()
#         )

#         return subjects

#     def _get_general_subject_questions(self, subject):
#         """
#         Get validation rows for one original
#         subject inside general.
#         """

#         subject_df = self.df[
#             (self.df["expected_service"] == "general")
#             & (self.df["subject"] == subject)
#         ].copy()

#         return subject_df

#     def _format_questions(self, df):
#         """
#         Convert validation questions into text
#         that can be passed to the LLM.
#         """

#         formatted_questions = []

#         for i, row in df.reset_index(drop=True).iterrows():

#             choices = "\n".join(
#                 f"- {choice}"
#                 for choice in row["choices"]
#             )

#             text = f"""
# QUESTION {i + 1}:
# {row["question"]}

# CHOICES:
# {choices}
# """

#             formatted_questions.append(text.strip())

#         return "\n\n".join(formatted_questions)

#     def inspect(self):

#         print("\n" + "=" * 60)
#         print("SPECIALIST SERVICES")
#         print("=" * 60)

#         services = self._get_specialist_services()

#         for service in services:

#             service_df = self._get_service_questions(
#                 service
#             )

#             print(
#                 f"{service}: "
#                 f"{len(service_df)} questions"
#             )

#         print("\n" + "=" * 60)
#         print("GENERAL SERVICE")
#         print("=" * 60)

#         general_subjects = (
#             self._get_general_subjects()
#         )

#         for subject in general_subjects:

#             subject_df = (
#                 self._get_general_subject_questions(
#                     subject
#                 )
#             )

#             print(
#                 f"{subject}: "
#                 f"{len(subject_df)} questions"
#             )


# if __name__ == "__main__":

#     extractor = ServiceConceptExtractor()

#     service_df = extractor._get_service_questions(
#         "abstract_algebra"
#     )

#     formatted = extractor._format_questions(
#         service_df
#     )

#     print(formatted)