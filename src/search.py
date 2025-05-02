import redis
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
import csv
from datetime import datetime
from time import time
import psutil



# Initialize models
# minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
# mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1") 
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

# Model configurations
MODEL_CONFIG = {
    "nomic": {
        "dim": 768,
        "get_embedding": lambda text: ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
    },
    "minilm": {
        "dim": 384,
        "get_embedding": lambda text: ollama.embeddings(model="all-minilm", prompt=text)["embedding"]
    },
    "mxbai": {
        "dim": 1024,
        "get_embedding": lambda text: ollama.embeddings(model="mxbai-embed-large", prompt=text)["embedding"]
    }
}

INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def log_to_csv(embedding_model, llm_model, prompt, response_time, response_length, prompt_length):
    """Log query details to CSV file"""
    file_exists = os.path.isfile('data_collection.csv')
    
    with open('data_collection.csv', 'a', newline='') as csvfile:
        fieldnames = ['timestamp', 'ram', 'database', 'embedding', 'llm', 'prompt', 'response_time_sec', 'response_length', 'prompt_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()

        mem = psutil.virtual_memory()
        primary_memory_size = mem.total/(1024**3)
            
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'ram': primary_memory_size,
            'database': 'redis',
            'embedding': embedding_model,
            'llm': llm_model,
            'prompt': prompt,
            'response_time_sec': response_time,
            'response_length': response_length,
            'prompt_length': prompt_length, 
        })

def get_embedding(text: str, embedding_model: str) -> list:
    try:
        return MODEL_CONFIG[embedding_model]["get_embedding"](text)
    except KeyError:
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"] # nomic-embed-text is default

def search_embeddings(query, embedding_model, top_k=3):
    try:
        # query_embedding = get_embedding(query, embedding_model)
        # query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

        # q = (
        #     Query("*=>[KNN {top_k} @embedding $vec AS vector_distance]")
        #     .sort_by("vector_distance")
        #     .return_fields("file", "page", "chunk", "vector_distance")
        #     .dialect(2)
        # )

        # results = redis_client.ft(INDEX_NAME).search(
        #     q, 
        #     query_params={"vec": query_vector},
        #     # params={"top_k": top_k}
        # )
        embedding = get_embedding(query, embedding_model)
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("file", "page", "chunk", "vector_distance")
            .dialect(2)
        )
        res = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
        )
        
        return [
            {
                "file": doc.file,
                "page": doc.page,
                "chunk": doc.chunk,
                "similarity": doc.vector_distance,
            }
            for doc in res.docs
        ]
    except Exception as e:
        print(e)
        return []

def generate_rag_response(query, context_results, embedding_model='ollama', llm_model="mistral:latest"):
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""
    print(prompt)
    # Generate response using Ollama
    ollama_response = ollama.chat(
        model=llm_model, messages=[{"role": "user", "content": prompt}]
    )
    return ollama_response["message"]["content"], len(prompt)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def interactive_search(embedding_model='nomic', llm_model="mistral:latest", query=None):
    prompt = 0
    while True:
        if query:
            prompt = 1
        else:
            query = input("\nEnter query (or 'exit'): ")
        
        if query.lower() == "exit":
            break

        start_time = time()
        results = search_embeddings(query, embedding_model)
        response, pl = generate_rag_response(query, results, llm_model)
        end_time = time()
        
        response_time = end_time - start_time
        response_length = len(response)
        
        print(f"\n🤖 Response ({embedding_model}):\n{response}")
        print(f"\n⏱️  Response time: {response_time:.2f} seconds")
        print(f"📏 Response length: {response_length} characters")
        
        log_to_csv(embedding_model, llm_model, query, response_time, response_length, pl)
        if prompt == 1:
            break

if __name__ == "__main__":
    interactive_search()