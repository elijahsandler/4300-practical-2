from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
import os
import pymupdf
import json
from time import time
from chromadb.config import Settings
import ollama


# Initialize models
sentence_transformers_all_minilm = SentenceTransformer("all-MiniLM-L6-v2")
sentence_transformers_all_mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
client = chromadb.Client()

VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        # q = (
        #     Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        #     .sort_by("vector_distance")
        #     .return_fields("id", "file", "page", "chunk", "vector_distance")
        #     .dialect(2)
        # )
        query_embedding = get_embedding(query)
        
        embedding = get_embedding(query_embedding)
        collection = client.get_collection(COLLECTION_NAME)
        
        # Perform the search
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5
        )

        for i, result in enumerate(results['documents']):
            print(f"Document {i+1}: {result} with distance: {results['distances'][i]}\n\n")
    

        # Transform results into the expected format
        top_results = [{
            "file": metadata['file'],
            "page": metadata['page'],
            "chunk": doc_chunk,
            "similarity": distance
                }
            for metadata, 
                doc_chunk, 
                distance in zip(results['metadatas'][0], 
                                results['documents'][0], 
                                results['distances'][0])
                ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


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
    if embedding_model == 'ollama':
        ollama_response = ollama.chat(
            model="mistral:latest", messages=[{"role": "user", "content": prompt}]
        )
        return ollama_response["message"]["content"]

    # Generate response using sentence_transformers_all_minilm (sentence-transformers/all-MiniLM-L6-v2)
    elif embedding_model == 'minilm':
        minilm_response = sentence_transformers_all_minilm.encode(query) # text
        return minilm_response

    # Generate response using sentence_transformers_all_mpnet (sentence-transformers/all-mpnet-base-v2)
    elif embedding_model == 'mpnet':
        mpnet_response = sentence_transformers_all_mpnet.encode(prompt) # sentence
        return mpnet_response

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

clear_terminal()

def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")
    print("Type 'clear' to clear terminal")
    while True:
        print("Pick a model out of... \n",
            "0: nomic-embed-text \n",
            "1: SentenceTransformer all-MiniLM-L6-v2 \n",
            "2: SentenceTransformer all-mpnet-base-v2")
        model_num = int(input("Pick model number: "))
        if model_num in (0, 1, 2):
            break
        else:
            "Please pick only 0, 1 or 2"
            
    if model_num == 0:
        embedding_model = "ollama"
    elif model_num == 1:
        embedding_model = "minilm"
    elif model_num ==2:
        embedding_model = "mpnet"

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break
        elif query.lower() == 'clear':
            clear_terminal()
            print("🔍 RAG Search Interface")
            print("Type 'exit' to quit")
            print("Type 'clear' to clear terminal")
        else: 
            # Search for relevant embeddings
            context_results = search_embeddings(query)

            # Generate RAG response
            response = generate_rag_response(query, context_results, embedding_model)

            print("\n--- Query ---")
            print(query)

            print("\n--- Response ---")
            print(response)
            print(response.strip(), '\n')


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
