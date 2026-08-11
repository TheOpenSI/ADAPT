import json
import ollama
from pathlib import Path


class ServiceDescriptionGenerator:
    """
    Generate one routing description for each service
    from its TP/FN/FP routing profile.
    """

    def __init__(
        self,
        services,
        model="qwen2.5:32b",
        results_dir="../data",
        temperature=0,
    ):
        self.services = services
        self.model = model
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature

    def _build_system_prompt(self):
        services_text = "\n".join(
            f"- {service}"
            for service in self.services
    )
        return f"""Write a concise semantic description for the target routing service.

AVAILABLE SERVICES:
{services_text}

Use the evidence as follows:
- correctly recognized examples → identify the core domain
- missed examples → identify legitimate domain concepts that may be underrepresented
- false matches → identify useful boundaries with overlapping available services

Write 40-60 words that:
- summarize 4-6 broad domain concepts
- abstract away from individual examples
- define the service by subject matter, not by difficulty, reasoning style, factuality, or computational complexity
- use accurate domain-specific terminology
- mention another available service only when a meaningful domain boundary exists

Desired structure:
1. core domain concepts
2. broader scope
3. optional domain-based boundary

Example structure:
"Stoichiometry, reaction kinetics, acid-base chemistry, equilibrium, and thermodynamics form the core of high school chemistry. The domain includes chemical reactions, quantitative relationships, solution chemistry, and experimental chemical principles. It differs from college_physics by focusing on substances, chemical transformations, and molecular interactions rather than general physical laws and phenomena."

Use the example only as a structural guide; do not copy its wording or subject matter.

Do not:
- list individual questions or cases
- invent services or unsupported topics
- use invalid labels
- define boundaries as factual vs reasoning, simple vs difficult, or calculation vs explanation
- mention routing, predictions, errors, TP/FN/FP, MMLU, datasets, or profiling
- begin with "This service" or "The service covers"

The general service represents domains not covered by an available specialist.

Return only the description."""

    def _build_user_prompt(self, profile):

        return f"""Target service:
{profile["service"]}

Correctly recognized examples:
{json.dumps(profile["true_positive_examples"], indent=2)}

Missed examples that belong to this service:
{json.dumps(profile["false_negative_examples"], indent=2)}

False matches from other services:
{json.dumps(profile["false_positive_examples"], indent=2)}

Using this evidence, write the routing description for the target service."""


    def _generate_description(self, profile):

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(profile),
                },
            ],
            options={
                "temperature": self.temperature
            },
        )

        return response["message"]["content"].strip()


    def _save_descriptions(self, descriptions):

        path = (
            self.results_dir
            / f"service_descriptions_{len(self.services) - 1}_services.json"
        )

        with open(path, "w") as f:
            json.dump(
                descriptions,
                f,
                indent=2,
            )

        print(
            f"[3] Service descriptions saved to {path}"
        )

        return path


    def generate_descriptions(self, profiles):
        print("\n" + "=" * 50)
        print("SERVICE DESCRIPTION GENERATOR")
        print("=" * 50)
        print(f"Model:       {self.model}")
        print(f"Temperature: {self.temperature}")
        print(f"Services:    {len(self.services)}")
        print(f"Output dir:  {self.results_dir}")
        print("=" * 50)

        print(
            f"[1] Generating descriptions "
            f"for {len(profiles)} routing labels"
            )

        descriptions = {}

        for profile in profiles:
            service = profile["service"]

            print(f"\n    Generating: {service}")

            description = self._generate_description(profile)

            descriptions[service] = description

            print(f"    Description: {description}")

        print(f"\n[2] Generated {len(descriptions)} descriptions")

        self._save_descriptions(descriptions)

        return descriptions