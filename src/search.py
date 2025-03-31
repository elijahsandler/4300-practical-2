import redis
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField


# Initialize models
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1") 
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_embedding(text: str, embedding_model: str) -> list:
    if embedding_model == "nomic":
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
    elif embedding_model == "minilm":
        return minilm_model.encode(text).tolist()
    elif embedding_model == "mxbai-embed-large":
        return mxbai_model.encode(text).tolist()
    else:
        raise ValueError(f"Unknown model: {embedding_model}")


def search_embeddings(query, embedding_model, top_k=3):
    query_embedding = get_embedding(query, embedding_model)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    # Search only embeddings from the selected model
    q = (
        Query(f"@embedding_model:{embedding_model}=>[KNN {top_k} @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance")
        .dialect(2)
    )

    results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vector})
    return [
        {
            "file": doc.file,
            "page": doc.page,
            "chunk": doc.chunk,
            "similarity": doc.vector_distance,
        }
        for doc in results.docs
    ]

# def search_embeddings(query, top_k=3):

#     query_embedding = get_embedding(query)

#     # Convert embedding to bytes for Redis search
#     query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

#     try:
#         # Construct the vector similarity search query
#         # Use a more standard RediSearch vector search syntax
#         # q = Query("*").sort_by("embedding", query_vector)

#         q = (
#             Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
#             .sort_by("vector_distance")
#             .return_fields("id", "file", "page", "chunk", "vector_distance")
#             .dialect(2)
#         )

#         # Perform the search
#         results = redis_client.ft(INDEX_NAME).search(
#             q, query_params={"vec": query_vector}
#         )

#         # Transform results into the expected format
#         top_results = [
#             {
#                 "file": result.file,
#                 "page": result.page,
#                 "chunk": result.chunk,
#                 "similarity": result.vector_distance,
#             }
#             for result in results.docs
#         ][:top_k]

#         # Print results for debugging
#         for result in top_results:
#             print(
#                 f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
#             )

#         return top_results

#     except Exception as e:
#         print(f"Search error: {e}")
#         return []


def generate_rag_response(query, context_results, embedding_model='ollama'):

    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    ollama_response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )
    return ollama_response["message"]["content"]


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_terminal()

def interactive_search():
    print("🔍 Choose Embedding Model:")
    print("1. nomic-embed-text (Ollama)")
    print("2. all-MiniLM-L6-v2 (SentenceTransformers)")
    print("3. mxbai-embed-large")

    choice = input("Enter model number (1/2/3): ")
    model_map = {"1": "nomic", "2": "minilm", "3": "mxbai-embed-large"}
    embedding_model = model_map.get(choice, "nomic")  # Default to nomic

    while True:
        query = input("\nEnter query (or 'exit'): ")
        if query.lower() == "exit":
            break

        results = search_embeddings(query, embedding_model)
        response = generate_rag_response(query, results)
        print(f"\n🤖 Response ({embedding_model}):\n{response}")


# def store_embedding(file, page, chunk, embedding):
#     """
#     Store an embedding in Redis using a hash with vector field.

#     Args:
#         file (str): Source file name
#         page (str): Page number
#         chunk (str): Chunk index
#         embedding (list): Embedding vector
#     """
#     key = f"{file}_page_{page}_chunk_{chunk}"
#     redis_client.hset(
#         key,
#         mapping={
#             "embedding": np.array(embedding, dtype=np.float32).tobytes(),
#             "file": file,
#             "page": page,
#             "chunk": chunk,
#         },
#     )


if __name__ == "__main__":
    interactive_search()
