import pymilvus
import numpy as np
import os
import pymupdf
import json
from time import time
import ollama

# Milvus client initialization
milvus_host = "localhost" 
milvus_port = "19530"  # Default port for Milvus
client = pymilvus.Milvus(host=milvus_host, port=milvus_port)

VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "COSINE" 

# Create Milvus collection to store embeddings
def create_milvus_collection():
    if COLLECTION_NAME in client.list_collections():
        print(f"Collection {COLLECTION_NAME} already exists.")
        return
    
    # Define the schema for the collection
    fields = [
        pymilvus.FieldSchema(name="embedding", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        pymilvus.FieldSchema(name="file", dtype=pymilvus.DataType.STRING),
        pymilvus.FieldSchema(name="page", dtype=pymilvus.DataType.STRING),
        pymilvus.FieldSchema(name="chunk", dtype=pymilvus.DataType.STRING),
    ]
    
    # Create the collection
    collection_schema = pymilvus.CollectionSchema(fields, description="Embedding collection")
    collection = pymilvus.Collection(name=COLLECTION_NAME, schema=collection_schema)
    print("Milvus collection created successfully.")

# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store the embedding in Milvus
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    # Prepare data for insertion
    entities = [
        np.array(embedding, dtype=np.float32),  # Embedding vector
        np.array([file], dtype="str"),  # File name
        np.array([page], dtype="str"),  # Page number
        np.array([chunk], dtype="str")  # Chunk of text
    ]
    
    # Insert the data into the Milvus collection
    client.insert(collection_name=COLLECTION_NAME, records=entities)
    print(f"Stored embedding for: {chunk}")

# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = pymupdf.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

# Process all PDF files in a given directory
def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def process_pys(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".py"):
            py_path = os.path.join(data_dir, file_name)
            with open(py_path, "r", encoding="utf-8") as file:
                code_text = file.read()
            
            chunks = split_text_into_chunks(code_text)
            for chunk_index, chunk in enumerate(chunks):
                embedding = get_embedding(chunk)
                store_embedding(
                    file=file_name,
                    page="1",  # Treat the entire file as one page
                    chunk=str(chunk),
                    embedding=embedding,
                )
            print(f" -----> Processed {file_name}")

def process_ipynbs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".ipynb"):
            ipynb_path = os.path.join(data_dir, file_name)
            with open(ipynb_path, "r", encoding="utf-8") as file:
                notebook_data = json.load(file)
            
            for page_num, cell in enumerate(notebook_data.get("cells", [])):
                if cell.get("cell_type") == "code":
                    cell_text = "\n".join(cell.get("source", []))
                    chunks = split_text_into_chunks(cell_text)
                    for chunk_index, chunk in enumerate(chunks):
                        embedding = get_embedding(chunk)
                        store_embedding(
                            file=file_name,
                            page=str(page_num + 1),  # Treat each cell as a page
                            chunk=str(chunk),
                            embedding=embedding,
                        )
            print(f" -----> Processed {file_name}")

# Query Milvus for similar documents
def query_milvus(query_text: str):
    embedding = get_embedding(query_text)
    
    # Perform similarity search in Milvus
    search_params = {"nprobe": 16}  # Number of probes to search for (higher is more accurate but slower)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_records=[np.array(embedding, dtype=np.float32)],
        top_k=5,  # Top 5 most similar results
        params=search_params
    )
    
    for i, result in enumerate(results[0]):
        print(f"Document {i+1}: {result.id} with distance: {result.distance}\n\n")

def main():
    create_milvus_collection()

    s = time()
    process_pdfs("./data/")
    process_pys("./data/")
    process_ipynbs("./data/")
    t = time() - s
    print(f"\n---Processed documents in {t} seconds---\n")

    query_milvus("What is the capital of France?")

if __name__ == "__main__":
    main()