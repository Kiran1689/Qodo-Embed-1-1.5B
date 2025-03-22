from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B", trust_remote_code=True)

snippets = [
    """def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1""",

    """def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]""",

    """def reverse_linked_list(head):
    prev = None
    current = head
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    return prev"""
]

query = "How to reverse a linked list in Python"
query_embedding = model.encode(query)
snippets_embeddings = model.encode(snippets)

similarities = cosine_similarity([query_embedding], snippets_embeddings)
most_similar_idx = np.argmax(similarities)

print("Most relevant code snippet:")
print(snippets[most_similar_idx])