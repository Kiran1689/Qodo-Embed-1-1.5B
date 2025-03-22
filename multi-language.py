import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B", trust_remote_code=True)

queries = [
    "How to define a function in Java",
    "How to create a list in Ruby"
]

snippets = [
    """public void greet() {
    System.out.println("Hello World");
}""",
    """my_list = ["apple", "banana", "cherry"]"""
]

query_embeddings = model.encode(queries)
snippet_embeddings = model.encode(snippets)
scores = cosine_similarity(query_embeddings, snippet_embeddings)

for i, query in enumerate(queries):
    best_match = snippets[np.argmax(scores[i])]
    print(f"Best match for '{query}':\n{best_match}\n")