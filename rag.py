import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B", trust_remote_code=True)

user_prompt = "Create a Flask route that accepts POST requests and returns JSON"
context_snippets = [
    """from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    return jsonify(data)
""",
    """from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/items")
def create_item(request: Request):
    return {"message": "Item received"}"""
]

prompt_embedding = model.encode(user_prompt)
context_embeddings = model.encode(context_snippets)
scores = cosine_similarity([prompt_embedding], context_embeddings)

best_match = context_snippets[np.argmax(scores)]
print("Context snippet to augment generation:")
print(best_match)