import weaviate
import numpy as np
import os
import pymupdf
import json
from time import time
import ollama
import weaviate.classes.config as wvcc
from weaviate.classes.query import MetadataQuery


client = weaviate.connect_to_local(
    port=8080,
    grpc_port=50051
    )

VECTOR_DIM = 768
COLLECTION_NAME = "EmbeddingCollection"

def get_weaviate_collection():
    try:        
        collection = client.collections.get(COLLECTION_NAME)        
        print("Using existing collection")   
    except:     
        collection = client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=wvcc.Configure.Vectorizer.text2vec_cohere(),
        generative_config=wvcc.Configure.Generative.cohere(),
        properties=[
            wvcc.Property(
                name="title",
                data_type=wvcc.DataType.TEXT
            )
        ]
    )


    print("Weaviate class created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the embedding in Weaviate
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    doc_id = f"{file}_page_{page}_chunk_{chunk}"
    metadata = {
        "uuid": doc_id,
        "file": file,
        "page": page,
        "chunk": chunk
    }

    # Store the data in Weaviate
    client.collections.get(COLLECTION_NAME).data.insert(metadata, vector=embedding)
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


# Query Weaviate for similar documents
def query_weaviate(query_text: str):
    embedding = get_embedding(query_text)
    collection = client.collections.get(COLLECTION_NAME)
    # Perform a query to Weaviate using the embedding
    result = collection.query.near_vector(near_vector=embedding,
                                          limit=5, 
                                          return_metadata=MetadataQuery(distance=True))


    for i, obj in enumerate(result.objects):
        # Access the 'properties' dictionary of each Object
        properties = obj.properties
        print(f"Document {i+1}: {properties['file']} (Page: {properties['page']}) - Chunk: {properties['chunk']}")



def main():
    get_weaviate_collection()

    s = time()
    process_pdfs("./data/")
    process_pys("./data/")
    process_ipynbs("./data/")
    t = time() - s
    print(f"\n---Processed documents in {t} seconds---\n")

    query_weaviate("What is the capital of France?")
    
    client.close()

if __name__ == "__main__":
    main()