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

# import weaviate
# import os
# import pymupdf
# import json
# from time import time
# import ollama
# from sentence_transformers import SentenceTransformer
# from weaviate.classes.config import Configure, Property, DataType

# # Initialize models
# minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
# mxbai_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# # Initialize Weaviate client
# try:
#     client = weaviate.connect_to_local(
#         host="localhost",
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

# def clear_weaviate_store():
#     print("Clearing existing Weaviate store...")
#     try:
#         client.collections.delete(COLLECTION_NAME)
#     except:
#         pass  # Collection doesn't exist
#     print("Weaviate store cleared.")

# def create_weaviate_collection(embedding_model):
#     dim = MODEL_CONFIG[embedding_model]["dim"]
#     clear_weaviate_store()
    
#     # Create a new collection with the appropriate configuration
#     client.collections.create(
#         name=COLLECTION_NAME,
#         properties=[
#             Property(name="file", data_type=DataType.TEXT),
#             Property(name="page", data_type=DataType.TEXT),
#             Property(name="chunk", data_type=DataType.TEXT),
#             Property(name="embedding_model", data_type=DataType.TEXT)
#         ],
#         vectorizer_config=None  # We're providing our own vectors
#     )
#     print(f"Collection created successfully for {embedding_model} (dim={dim}).")

# def get_embedding(text: str, embedding_model: str) -> list:
#     return MODEL_CONFIG[embedding_model]["get_embedding"](text)

# def store_embedding(file: str, page: str, chunk: str, embedding: list, embedding_model: str):
#     doc_id = f"{file}_page_{page}_chunk_{chunk}"
#     data_object = {
#         "file": file,
#         "page": page,
#         "chunk": chunk,
#         "embedding_model": embedding_model
#     }

#     # Store the data in Weaviate with our custom embedding
#     client.collections.get(COLLECTION_NAME).data.insert(
#         properties=data_object,
#         vector=embedding,
#         uuid=doc_id
#     )

# def extract_text_from_pdf(pdf_path):
#     doc = pymupdf.open(pdf_path)
#     return [(page_num, page.get_text()) for page_num, page in enumerate(doc)]

# def split_text_into_chunks(text, chunk_size=300, overlap=50):
#     words = text.split()
#     return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# def process_files(data_dir, file_extension, process_func, embedding_model):
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(file_extension):
#             file_path = os.path.join(data_dir, file_name)
#             for page_num, text in process_func(file_path):
#                 chunks = split_text_into_chunks(text)
#                 for chunk in chunks:
#                     embedding = get_embedding(chunk, embedding_model)
#                     store_embedding(
#                         file=file_name,
#                         page=str(page_num),
#                         chunk=chunk,
#                         embedding=embedding,
#                         embedding_model=embedding_model
#                     )
#             print(f"Processed {file_name} with {embedding_model}")

# def process_pdfs(data_dir, embedding_model):
#     process_files(data_dir, ".pdf", extract_text_from_pdf, embedding_model)

# def process_pys(data_dir, embedding_model):
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".py"):
#             with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
#                 chunks = split_text_into_chunks(file.read())
#                 for chunk in chunks:
#                     embedding = get_embedding(chunk, embedding_model)
#                     store_embedding(
#                         file=file_name,
#                         page="1",
#                         chunk=chunk,
#                         embedding=embedding,
#                         embedding_model=embedding_model
#                     )
#             print(f"Processed {file_name} with {embedding_model}")

# def process_ipynbs(data_dir, embedding_model):
#     for file_name in os.listdir(data_dir):
#         if file_name.endswith(".ipynb"):
#             with open(os.path.join(data_dir, file_name), "r", encoding="utf-8") as file:
#                 notebook = json.load(file)
#                 for page_num, cell in enumerate(notebook.get("cells", [])):
#                     if cell.get("cell_type") == "code":
#                         chunks = split_text_into_chunks("\n".join(cell.get("source", [])))
#                         for chunk in chunks:
#                             embedding = get_embedding(chunk, embedding_model)
#                             store_embedding(
#                                 file=file_name,
#                                 page=str(page_num + 1),
#                                 chunk=chunk,
#                                 embedding=embedding,
#                                 embedding_model=embedding_model
#                             )
#             print(f"Processed {file_name} with {embedding_model}")

# def query_weaviate(query_text: str, embedding_model: str):
#     embedding = get_embedding(query_text, embedding_model)
#     collection = client.collections.get(COLLECTION_NAME)
    
#     result = collection.query.near_vector(
#         near_vector=embedding,
#         limit=5,
#         return_metadata=["distance"]
#     )

#     for obj in result.objects:
#         properties = obj.properties
#         distance = obj.metadata.distance
#         similarity = 1 - distance  # Convert distance to similarity
#         print(f"{properties['file']} (page {properties['page']})\n{properties['chunk'][:200]}...\nSimilarity: {similarity:.4f}\n")

# def select_embedding_model():
#     print("🔍 Choose Embedding Model:")
#     print("1. nomic-embed-text (Ollama)")
#     print("2. all-MiniLM-L6-v2 (SentenceTransformers)")
#     print("3. mxbai-embed-large (SentenceTransformers)")
#     choice = input("Enter model number (1/2/3): ")
#     return {"1": "nomic", "2": "minilm", "3": "mxbai"}.get(choice, "nomic")

# def main():
#     embedding_model = select_embedding_model()
#     print(f"\nUsing {embedding_model} embedding model\n")
    
#     create_weaviate_collection(embedding_model)

#     start_time = time()
#     process_pdfs("./data/", embedding_model)
#     process_pys("./data/", embedding_model)
#     process_ipynbs("./data/", embedding_model)
    
#     print(f"\nProcessing completed in {time() - start_time:.2f} seconds")
#     query_weaviate("What is the capital of France?", embedding_model)
    
#     client.close()

# if __name__ == "__main__":
#     main()