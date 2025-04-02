## DS 4300 Example - from docs
import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import pymupdf
import json
from time import time
from sentence_transformers import SentenceTransformer

# Initialize models
# minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
# mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
redis_client = redis.Redis(host="localhost", port=6380, db=0)

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

def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")

def create_hnsw_index(embedding_model):
    dim = MODEL_CONFIG[embedding_model]["dim"]
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME}")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {dim} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print(f"Index created successfully for {embedding_model} (dim={dim}).")

def get_embedding(text: str, embedding_model: str) -> list:
    return MODEL_CONFIG[embedding_model]["get_embedding"](text)

def store_embedding(file: str, page: str, chunk: str, embedding: list, embedding_model: str):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
            "embedding_model": embedding_model
        },
    )

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

def split_text_into_chunks(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

def process_files(data_dir, file_extension, process_func, embedding_model):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(file_extension):
            file_path = os.path.join(data_dir, file_name)
            for page_num, text in process_func(file_path):
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = get_embedding(chunk, embedding_model)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=chunk,
                        embedding=embedding,
                        embedding_model=embedding_model
                    )
            print(f"Processed {file_name} with {embedding_model}")

def process_pdfs(data_dir, embedding_model):
    process_files(data_dir, ".pdf", extract_text_from_pdf, embedding_model)

def process_pys(data_dir, embedding_model):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".py"):
            with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
                chunks = split_text_into_chunks(file.read())
                for chunk in chunks:
                    embedding = get_embedding(chunk, embedding_model)
                    store_embedding(
                        file=file_name,
                        page="1",
                        chunk=chunk,
                        embedding=embedding,
                        embedding_model=embedding_model
                    )
            print(f"Processed {file_name} with {embedding_model}")

def process_ipynbs(data_dir, embedding_model):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".ipynb"):
            with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
                notebook = json.load(file)
                for page_num, cell in enumerate(notebook.get("cells", [])):
                    if cell.get("cell_type") == "code":
                        chunks = split_text_into_chunks("\n".join(cell.get("source", [])))
                        for chunk in chunks:
                            embedding = get_embedding(chunk, embedding_model)
                            store_embedding(
                                file=file_name,
                                page=str(page_num + 1),
                                chunk=chunk,
                                embedding=embedding,
                                embedding_model=embedding_model
                            )
            print(f"Processed {file_name} with {embedding_model}")

def query_redis(query_text: str, embedding_model: str):
    embedding = get_embedding(query_text, embedding_model)
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("file", "page", "chunk", "vector_distance")
        .dialect(2)
    )
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    for doc in res.docs:
        print(f"{doc.file} (page {doc.page})\n{doc.chunk[:200]}...\nSimilarity: {doc.vector_distance}\n")

def select_embedding_model():
    print("🔍 Choose Embedding Model:")
    print("1. nomic-embed-text (Ollama)")
    print("2. all-MiniLM-L6-v2 (SentenceTransformers)")
    print("3. mxbai-embed-large (SentenceTransformers)")
    choice = input("Enter model number (1/2/3): ")
    return {"1": "nomic", "2": "minilm", "3": "mxbai"}.get(choice, "nomic")

def main(embedding_model=None):
    if not embedding_model:
        embedding_model = select_embedding_model()
    else:
        embedding_model = {"1": "nomic", "2": "minilm", "3": "mxbai"}.get(embedding_model, "nomic")
    print(f"\nUsing {embedding_model} embedding model\n")
    
    clear_redis_store()
    create_hnsw_index(embedding_model)

    start_time = time()
    process_pdfs("./data/", embedding_model)
    process_pys("./data/", embedding_model)
    process_ipynbs("./data/", embedding_model)
    
    print(f"\nProcessing completed in {time() - start_time:.2f} seconds")
    query_redis("What is the capital of France?", embedding_model)

if __name__ == "__main__":
    main()