from sentence_transformers import SentenceTransformer
import weaviate
import os
from time import time
import ollama
from weaviate.classes.query import MetadataQuery


# Initialize models
sentence_transformers_all_minilm = SentenceTransformer("all-MiniLM-L6-v2")
sentence_transformers_all_mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051
    )

VECTOR_DIM = 768
COLLECTION_NAME = "EmbeddingCollection"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(query, top_k=3):
    try:
        # Construct the vector similarity search query
        embedding = get_embedding(query)
        collection = client.collections.get(COLLECTION_NAME)
        # Perform a query to Weaviate using the embedding
        results = collection.query.near_vector(near_vector=embedding,
                                          limit=5, 
                                          return_metadata=MetadataQuery(distance=True))

        for i, obj in enumerate(results.objects):
            # Accessing the properties of each object
            file = obj.properties['file']
            chunk = obj.properties['chunk']
            page = obj.properties['page']
            
            # Getting the distance for the current object
            distance = obj.metadata.distance 

            # Printing the document details with the distance
            print(f"Document {i+1}: {file} - Page {page} - Chunk: {chunk} with distance: {distance}\n\n")
    

        # Transform results into the expected format 
        top_results = [{
            "file": obj.properties['file'],
            "page": obj.properties['page'],
            "chunk": obj.properties['chunk'],
            "similarity": obj.metadata.distance  # You can add similarity logic if needed
            }
        for obj in results.objects][:top_k]

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
            # print(response)
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

# from sentence_transformers import SentenceTransformer
# import weaviate
# import os
# import ollama
# import csv
# from datetime import datetime
# from time import time
# from weaviate.classes.query import MetadataQuery

# # Initialize models
# minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
# mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# # Initialize Weaviate client
# try:
#     client = weaviate.connect_to_local(
#         port=8080,
#         grpc_port=50051
#     )
# except Exception as e:
#     print(f"Failed to connect to Weaviate: {e}")
#     exit(1)

# # Model configurations
# MODEL_CONFIG = {
#     "nomic": {
#         "dim": 768,
#         "get_embedding": lambda text: ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
#     },
#     "minilm": {
#         "dim": 384,
#         "get_embedding": lambda text: minilm_model.encode(text).tolist()
#     },
#     "mxbai": {
#         "dim": 1024,
#         "get_embedding": lambda text: mxbai_model.encode(text).tolist()
#     }
# }

# COLLECTION_NAME = "EmbeddingCollection"

# def log_to_csv(embedding_model, prompt, response_time, response_length):
#     """Log query details to CSV file"""
#     file_exists = os.path.isfile('data_collection.csv')
    
#     with open('data_collection.csv', 'a', newline='') as csvfile:
#         fieldnames = ['timestamp', 'database', 'embedding', 'prompt', 'response_time_sec', 'response_length']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
#         if not file_exists:
#             writer.writeheader()
            
#         writer.writerow({
#             'timestamp': datetime.now().isoformat(),
#             'database': 'weaviate',
#             'embedding': embedding_model,
#             'prompt': prompt,
#             'response_time_sec': response_time,
#             'response_length': response_length
#         })

# def get_embedding(text: str, embedding_model: str) -> list:
#     try:
#         return MODEL_CONFIG[embedding_model]["get_embedding"](text)
#     except KeyError:
#         return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

# def search_embeddings(query, embedding_model, top_k=3):
#     try:
#         query_embedding = get_embedding(query, embedding_model)
#         collection = client.collections.get(COLLECTION_NAME)
        
#         results = collection.query.near_vector(
#             near_vector=query_embedding,
#             limit=top_k,
#             return_metadata=MetadataQuery(distance=True)
#         )
        
#         return [
#             {
#                 "file": obj.properties['file'],
#                 "page": obj.properties['page'],
#                 "chunk": obj.properties['chunk'],
#                 "similarity": 1 - obj.metadata.distance  # Convert distance to similarity
#             }
#             for obj in results.objects
#         ]
#     except Exception as e:
#         print(f"Search error: {e}")
#         return []

# def generate_rag_response(query, context_results, embedding_model='nomic'):
#     # Prepare context string
#     context_str = "\n".join(
#         [
#             f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
#             f"with similarity {float(result.get('similarity', 0)):.2f}"
#             for result in context_results
#         ]
#     )

#     # Construct prompt with context
#     prompt = f"""You are a helpful AI assistant. 
#     Use the following context to answer the query as accurately as possible. If the context is 
#     not relevant to the query, say 'I don't know'.

# Context:
# {context_str}

# Query: {query}

# Answer:"""

#     # Generate response using Ollama
#     ollama_response = ollama.chat(
#         model="mistral:latest", messages=[{"role": "user", "content": prompt}]
#     )
#     return ollama_response["message"]["content"]

# def clear_terminal():
#     os.system('cls' if os.name == 'nt' else 'clear')

# def interactive_search():
#     clear_terminal()
#     print("🔍 Choose Embedding Model:")
#     print("1. nomic-embed-text (Ollama)")
#     print("2. all-MiniLM-L6-v2 (SentenceTransformers)")
#     print("3. mxbai-embed-large (SentenceTransformers)")

#     choice = input("Enter model number (1/2/3): ")
#     model_map = {"1": "nomic", "2": "minilm", "3": "mxbai"}
#     embedding_model = model_map.get(choice, "nomic")

#     while True:
#         query = input("\nEnter query (or 'exit'): ")
#         if query.lower() == "exit":
#             break

#         start_time = time()
#         results = search_embeddings(query, embedding_model)
#         response = generate_rag_response(query, results, embedding_model)
#         end_time = time()
        
#         response_time = end_time - start_time
#         response_length = len(response)
        
#         print(f"\n🤖 Response ({embedding_model}):\n{response}")
#         print(f"\n⏱️  Response time: {response_time:.2f} seconds")
#         print(f"📏 Response length: {response_length} characters")
        
#         log_to_csv(embedding_model, query, response_time, response_length)
    
#     client.close()

# if __name__ == "__main__":
#     interactive_search()