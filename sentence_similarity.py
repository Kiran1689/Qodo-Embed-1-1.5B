from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the model
model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B", trust_remote_code=True)

# Source sentence and comparison list
source_sentence = "That is a very happy person"
sentences_to_compare = [
    "That is a happy person",
    "That is a happy dog",
    "Today is a sunny day",
    "The man is joyful and smiling"
]

# Encode source and comparison sentences
source_embedding = model.encode([source_sentence])
comparison_embeddings = model.encode(sentences_to_compare)

# Compute cosine similarity (returns a 2D array)
similarity_scores = cosine_similarity(source_embedding, comparison_embeddings)[0]

# Find most similar sentence
most_similar_idx = int(np.argmax(similarity_scores))
most_similar_sentence = sentences_to_compare[most_similar_idx]
similarity_score = similarity_scores[most_similar_idx]

# Print results
print(f"Source Sentence: \"{source_sentence}\"")
print(f"Most Similar Sentence: \"{most_similar_sentence}\"")
print(f"Similarity Score: {similarity_score:.4f}")