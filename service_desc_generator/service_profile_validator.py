import json
import ollama
import pandas as pd

class ServiceProfileValidator:

    def __init__(
        self,
        validation_path,
        profile_path,
        model="llama3.1:70b",
        verbose=True,
    ):
        self.validation_path = validation_path
        self.profile_path = profile_path
        self.model = model
        self.verbose = verbose

        self.client = ollama.Client(
            host="http://localhost:11435"
        )

        self.df = pd.read_json(self.validation_path)

        with open(self.profile_path) as f:
            self.profiles = json.load(f)

    def _log(self, message):
        if self.verbose:
            print(message)


    def _group_by_subject(self):
        return {subject: group.copy() for subject, group in self.df.groupby("subject")}


    def _format_questions(self, df):
        formatted_questions = []

        for i, row in (df.reset_index(drop=True).iterrows()):
            choices = "\n".join(f"- {choice}" for choice in row["choices"])

            formatted_questions.append(
                f"Question {i+1}:\n"
                f"{row['question']}\n"
                f"Choices:\n{choices}"
            )
        return "\n\n".join(formatted_questions)
    

    def _build_validation_prompt(
        self,
        subject,
        formatted_questions,
        profile,
    ):
        return f"""
You will receive two inputs:

(1) Validation questions from an academic specialist.

(2) A generated service profile derived from those questions.

Your task is to judge whether the service profile is consistent with the validation questions.

Field Definitions

- "function": A summary of the specialist's expertise.
- "tags": Technical concepts, theories, methods, structures, or terminology represented by the questions.

Important Rules

- Do not invent or assume information.
- Base your judgement only on the given validation questions.
- Every tag must be supported by one or more questions.
- The function must accurately summarize the expertise represented by the questions.
- Reject profiles that introduce unsupported concepts, theories, laws, doctrines, methods, or areas of expertise.
- Reject profiles that contradict the validation questions.

VALIDATION SUBJECT

{subject}

VALIDATION QUESTIONS

{formatted_questions}

SERVICE PROFILE

{json.dumps(profile, indent=2)}

Output only one word:

true

or

false
""".strip()
    

    def validate_subject(
            self,
            subject,
            df,
    ):

        profile = self.profiles[subject]

        formatted_questions = (
            self._format_questions(df)
        )

        prompt = self._build_validation_prompt(
            subject,
            formatted_questions,
            profile,
        )

        self._log("\n" + "=" * 80)
        self._log(f"[VALIDATE] {subject}: {len(df)} questions")

        self._log("\n===== QUESTIONS SENT TO LLM =====")
        self._log(formatted_questions)

        self._log("\n===== PROFILE SENT TO LLM =====")
        self._log(json.dumps(profile, indent=2))

        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={"temperature": 0},
        )

        result = (
            response.message.content
            .strip()
            .lower()
        )

        self._log(f"[RESULT] {subject}: {result}")

        if result not in ["true", "false"]:
            self._log(f"[WARNING] Unexpected response: {result}")

        return result == "true"


    def validate_all(
            self,
            output_path="service_profile_validation.json"
        ):

        groups = self._group_by_subject()

        results = {}

        for subject, df in groups.items():
            try:
                valid = self.validate_subject(subject, df)
                results[subject] = {"valid": valid}

            except Exception as e:
                results[subject] = {"valid": False, "error": str(e)}

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self._log(f"\n[DONE] Saved to {output_path}")

        return results



if __name__ == "__main__":

    validator = ServiceProfileValidator(
        validation_path="../service_desc_generator/dataset/validation_service_all.json",
        profile_path="../service_desc_generator/dataset/service_profiles.json",
    )

    validator.validate_all(
        output_path="../service_desc_generator/dataset/service_profile_validation.json"
    )