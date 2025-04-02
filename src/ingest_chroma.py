import chromadb
import os
import ollama
import pymupdf
import json
from time import time
from sentence_transformers import SentenceTransformer

# Initialize models
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1") 
client = chromadb.PersistentClient(path="./chroma_db")

# pull required Ollama models
required_models = ["nomic-embed-text", "mxbai-embed-large"]
for model in required_models:
    try:
        ollama.show(model)
    except:
        print(f"Pulling Ollama model: {model}")
        ollama.pull(model)

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
        "get_embedding": lambda text: minilm_model.encode(text).tolist()  # Using SentenceTransformer directly
    },
    "mxbai": {
        "dim": 1024,
        "collection_name": "mxbai_collection",
        "get_embedding": lambda text: ollama.embeddings(model="mxbai-embed-large", prompt=text)["embedding"]
    }
}

def initialize_collections():
    for model in MODEL_CONFIG.values():
        try:
            client.delete_collection(model["collection_name"])
        except ValueError:
            pass
        client.create_collection(
            name=model["collection_name"],
            metadata={"hnsw:space": "cosine"}
        )

def get_collection(embedding_model):
    collection_name = MODEL_CONFIG[embedding_model]["collection_name"]
    return client.get_collection(collection_name)

def get_embedding(text: str, embedding_model: str) -> list:
    try:
        return MODEL_CONFIG[embedding_model]["get_embedding"](text)
    except KeyError:
        return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

def store_embedding(file: str, page: str, chunk: str, embedding: list, embedding_model: str):
    collection = get_collection(embedding_model)
    doc_id = f"{file}_page_{page}_chunk_{hash(chunk)}"
    metadata = {
        "file": file,
        "page": page,
        "chunk": chunk[:500],
        "embedding_model": embedding_model
    }
    collection.add(
        documents=[chunk],
        metadatas=[metadata],
        embeddings=[embedding],
        ids=[doc_id]
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

def query_collection(query_text: str, embedding_model: str):
    embedding = get_embedding(query_text, embedding_model)
    collection = get_collection(embedding_model)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    print(f"\nResults for: '{query_text}'\n")
    for doc, meta, dist in zip(results['documents'][0], 
                             results['metadatas'][0], 
                             results['distances'][0]):
        print(f"File: {meta['file']} (page {meta['page']})")
        print(f"Similarity: {1 - dist:.4f}")
        print(f"Content: {doc[:200]}...\n")

def select_embedding_model():
    print("🔍 Choose Embedding Model:")
    print("1. nomic-embed-text (Ollama)")
    print("2. all-MiniLM-L6-v2 (SentenceTransformers)")
    print("3. mxbai-embed-large (Ollama)")
    choice = input("Enter model number (1/2/3): ")
    return {"1": "nomic", "2": "minilm", "3": "mxbai"}.get(choice, "nomic")

def main():
    embedding_model = select_embedding_model()
    print(f"\nUsing {embedding_model} embedding model\n")
    initialize_collections()
    start_time = time()
    process_pdfs("./data/", embedding_model)
    process_pys("./data/", embedding_model)
    process_ipynbs("./data/", embedding_model)
    
    print(f"\nProcessing completed in {time() - start_time:.2f} seconds")
    
if __name__ == "__main__":
    main()