import chromadb
import os
import ollama
# from sentence_transformers import SentenceTransformer
from time import time
from datetime import datetime
import csv
import psutil

# Initialize models
# minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
# mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1") 
client = chromadb.PersistentClient(path="./chroma_db")

# Model configurations
MODEL_CONFIG = {
    "nomic": {
        "dim": 768,
        "collection_name": "nomic_collection",
        "get_embedding": lambda text: ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
    },
    "minilm": {
        "dim": 384,
        "collection_name": "minilm_collection",
        "get_embedding": lambda text: ollama.embeddings(model="all-minilm", prompt=text)["embedding"]
    },
    "mxbai": {
        "dim": 1024,
        "collection_name": "mxbai_collection",
        "get_embedding": lambda text: ollama.embeddings(model="mxbai-embed-large", prompt=text)["embedding"]
    }
}

DISTANCE_METRIC = "cosine"

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
            'database': 'chroma',
            'embedding': embedding_model,
            'llm': llm_model,
            'prompt': prompt,
            'response_time_sec': response_time,
            'response_length': response_length,
            'prompt_length': prompt_length, 
        })

def get_collection(embedding_model):
    """Get or create the appropriate collection for the embedding model"""
    collection_name = MODEL_CONFIG[embedding_model]["collection_name"]
    try:
        return client.get_collection(collection_name)
    except:
        return client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC}
        )

def get_embedding(text: str, embedding_model: str) -> list:
    try:
        return MODEL_CONFIG[embedding_model]["get_embedding"](text)
    except KeyError:
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

def search_embeddings(query, embedding_model, top_k=3): # k=3 or 5 ?
    try:
        query_embedding = get_embedding(query, embedding_model)
        collection = get_collection(embedding_model)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        return [
            {
                "file": meta['file'],
                "page": meta['page'],
                "chunk": meta['chunk'],
                "similarity": 1 - dist  # Convert distance to similarity
            }
            for meta, dist in zip(results['metadatas'][0], 
                               results['distances'][0])
        ]
    except Exception as e:
        print(f"Search error: {e}")
        return []

def generate_rag_response(query, context_results, embedding_model='nomic', llm_model="mistral:latest"):
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

def interactive_search(embedding_model=None, llm_model="mistral:latest", query=None):
    clear_terminal()

    if not embedding_model:
        embedding_model = model_map.get(choice, "nomic")
        print("🔍 Choose Embedding Model:")
        print("1. nomic-embed-text (Ollama)")
        print("2. all-MiniLM-L6-v2 (SentenceTransformers)")
        print("3. mxbai-embed-large")

        choice = input("Enter model number (1/2/3): ")
    
    model_map = {"1": "nomic", "2": "minilm", "3": "mxbai-embed-large"}
    
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
        response, pl = generate_rag_response(query, results, embedding_model, llm_model)
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