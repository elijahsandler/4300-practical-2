# import chromadb
# import numpy as np
# import os
# import pymupdf
# import json
# from time import time
# from chromadb.config import Settings
# import ollama
# from sentence_transformers import SentenceTransformer


# # Initialize Chroma client with in-memory setup
# client = chromadb.PersistentClient(path="./chroma_db")

# VECTOR_DIM = 768
# COLLECTION_NAME = "embedding_collection"
# DISTANCE_METRIC = "cosine"

# # Create a Chroma collection to store embeddings
# def create_chroma_collection():
#     collection = client.create_collection(COLLECTION_NAME)
#     print("Chroma collection created successfully.")


# # Generate an embedding using nomic-embed-text
# def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
#     response = ollama.embeddings(model=model, prompt=text)
#     return response["embedding"]


# # Store the embedding in Chroma
# def store_embedding(file: str, page: str, chunk: str, embedding: list):
#     collection = client.get_collection(COLLECTION_NAME)
#     doc_id = f"{file}_page_{page}_chunk_{chunk}"
#     metadata = {"file": file, "page": page, "chunk": chunk}
#     collection.add(
#         documents=[chunk],
#         metadatas=[metadata],
#         embeddings=[embedding],
#         ids=[doc_id],
#     )
#     # print(f"Stored embedding for: {chunk}")


# # Extract the text from a PDF by page
# def extract_text_from_pdf(pdf_path):
#     """Extract text from a PDF file."""
#     doc = pymupdf.open(pdf_path)
#     text_by_page = []
#     for page_num, page in enumerate(doc):
#         text_by_page.append((page_num, page.get_text()))
#     return text_by_page


# # Split the text into chunks with overlap
# def split_text_into_chunks(text, chunk_size=300, overlap=50):
#     """Split text into chunks of approximately chunk_size words with overlap."""
#     words = text.split()
#     chunks = []
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = " ".join(words[i : i + chunk_size])
#         chunks.append(chunk)
#     return chunks


# # Process all PDF files in a given directory
# def process_pdfs(data_dir):
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".pdf"):
#             pdf_path = os.path.join(data_dir, file_name)
#             text_by_page = extract_text_from_pdf(pdf_path)
#             for page_num, text in text_by_page:
#                 chunks = split_text_into_chunks(text)
#                 for chunk_index, chunk in enumerate(chunks):
#                     embedding = get_embedding(chunk)
#                     store_embedding(
#                         file=file_name,
#                         page=str(page_num),
#                         chunk=str(chunk),
#                         embedding=embedding,
#                     )
#             print(f" -----> Processed {file_name}")


# def process_pys(data_dir):
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".py"):
#             py_path = os.path.join(data_dir, file_name)
#             with open(py_path, "r", encoding="utf-8") as file:
#                 code_text = file.read()
            
#             chunks = split_text_into_chunks(code_text)
#             for chunk_index, chunk in enumerate(chunks):
#                 embedding = get_embedding(chunk)
#                 store_embedding(
#                     file=file_name,
#                     page="1",  # Treat the entire file as one page
#                     chunk=str(chunk),
#                     embedding=embedding,
#                 )
#             print(f" -----> Processed {file_name}")


# def process_ipynbs(data_dir):
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".ipynb"):
#             ipynb_path = os.path.join(data_dir, file_name)
#             with open(ipynb_path, "r", encoding="utf-8") as file:
#                 notebook_data = json.load(file)
            
#             for page_num, cell in enumerate(notebook_data.get("cells", [])):
#                 if cell.get("cell_type") == "code":
#                     cell_text = "\n".join(cell.get("source", []))
#                     chunks = split_text_into_chunks(cell_text)
#                     for chunk_index, chunk in enumerate(chunks):
#                         embedding = get_embedding(chunk)
#                         store_embedding(
#                             file=file_name,
#                             page=str(page_num + 1),  # Treat each cell as a page
#                             chunk=str(chunk),
#                             embedding=embedding,
#                         )
#             print(f" -----> Processed {file_name}")


# # Query Chroma for similar documents
# def query_chroma(query_text: str):
#     embedding = get_embedding(query_text)
#     collection = client.get_collection(COLLECTION_NAME)
#     results = collection.query(
#         query_embeddings=[embedding],
#         n_results=5
#     )
    
#     for i, result in enumerate(results['documents']):
#         print(f"Document {i+1}: {result} with distance: {results['distances'][i]}\n\n")
    

# def main():
#     create_chroma_collection()

#     s = time()
#     process_pdfs("./data/")
#     process_pys("./data/")
#     process_ipynbs("./data/")
#     t = time() - s
#     print(f"\n---Processed documents in {t} seconds---\n")

#     query_chroma("What is the capital of France?")


# if __name__ == "__main__":
#     main()

import chromadb
import numpy as np
import os
import pymupdf
import json
from time import time
from chromadb.config import Settings
import ollama
from sentence_transformers import SentenceTransformer

# Initialize models
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Initialize Chroma client with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Model configurations
MODEL_CONFIG = {
    "nomic": {
        "dim": 768,
        "get_embedding": lambda text: ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
    },
    "minilm": {
        "dim": 384,
        "get_embedding": lambda text: minilm_model.encode(text).tolist()
    },
    "mxbai": {
        "dim": 1024,
        "get_embedding": lambda text: mxbai_model.encode(text).tolist()
    }
}

COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "cosine"

def clear_chroma_store():
    print("Clearing existing Chroma store...")
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass  # Collection doesn't exist
    print("Chroma store cleared.")

def create_chroma_collection(embedding_model):
    dim = MODEL_CONFIG[embedding_model]["dim"]
    clear_chroma_store()
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC}
    )
    print(f"Collection created successfully for {embedding_model} (dim={dim}).")
    return collection

def get_embedding(text: str, embedding_model: str) -> list:
    return MODEL_CONFIG[embedding_model]["get_embedding"](text)

def store_embedding(file: str, page: str, chunk: str, embedding: list, embedding_model: str):
    collection = client.get_collection(COLLECTION_NAME)
    doc_id = f"{file}_page_{page}_chunk_{chunk}"
    metadata = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "embedding_model": embedding_model
    }
    collection.add(
        documents=[chunk],
        metadatas=[metadata],
        embeddings=[embedding],
        ids=[doc_id],
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

def query_chroma(query_text: str, embedding_model: str):
    embedding = get_embedding(query_text, embedding_model)
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], 
                                          results['metadatas'][0], 
                                          results['distances'][0])):
        print(f"{meta['file']} (page {meta['page']})\n{doc[:200]}...\nSimilarity: {1 - dist:.4f}\n")

def select_embedding_model():
    print("🔍 Choose Embedding Model:")
    print("1. nomic-embed-text (Ollama)")
    print("2. all-MiniLM-L6-v2 (SentenceTransformers)")
    print("3. mxbai-embed-large (SentenceTransformers)")
    choice = input("Enter model number (1/2/3): ")
    return {"1": "nomic", "2": "minilm", "3": "mxbai"}.get(choice, "nomic")

def main():
    embedding_model = select_embedding_model()
    print(f"\nUsing {embedding_model} embedding model\n")
    
    create_chroma_collection(embedding_model)

    start_time = time()
    process_pdfs("./data/", embedding_model)
    process_pys("./data/", embedding_model)
    process_ipynbs("./data/", embedding_model)
    
    print(f"\nProcessing completed in {time() - start_time:.2f} seconds")
    
    while True:
        query_text = input("\nEnter your query (or 'exit' to exit): ")
        if query_text.lower() == 'exit':
            break
        query_chroma(query_text, embedding_model)

if __name__ == "__main__":
    main()