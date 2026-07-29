
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

prompt = "Let p = (1, 2, 5, 4)(2, 3) in S_5 . Find the index of <p> in S_5.?    A. 8    B. 2    C. 24    D. 120"
services = {
    "anatomy": "Answer questions about human anatomy, including but not limited to the structure, organisation, and physical features of body systems, organs, and tissues. Focused on normal structure rather than disease or treatment.",
    "abstract_algebra": "Answer questions about abstract algebra, including but not limited to groups, rings, fields, and other algebraic structures and their properties.",
    "astronomy": "Answer questions about astronomy, including but not limited to celestial objects, the solar system, stars, galaxies, and the physical processes of the universe."

}

prompt_embedding = model.encode(
    prompt, normalize_embeddings=True)

print(prompt_embedding.shape)

service_names = list(services.keys())
services_text = [f"{name}:{desc}" for name, desc in services.items()]
services_text_embedding = model.encode(
    services_text,
    normalize_embeddings=True
)

print(services_text_embedding.shape)

range(len(prompt_embedding))


# [3,384][,384]
def cosine_similarity(prompt_embedding,services_text_embedding):
    numerator = services_text_embedding @ prompt_embedding
    print("numerator",numerator)
    query_length = np.linalg.norm(prompt_embedding)
    print("query_length",query_length)
    service_lengths = np.linalg.norm(services_text_embedding, axis=1)
    print("service_lengths",service_lengths)
    denominator = service_lengths * query_length
    print("denominator",denominator)
    
    scores = numerator / denominator
    print("scores", scores)
    best_index = scores.argmax()
    predicted_service = service_names[best_index]

    return predicted_service


cosine_similarity(prompt_embedding,services_text_embedding)
