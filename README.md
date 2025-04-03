# DS4300 - Practical 2
Ensure all libraries in `requirements.txt` are installed. If they are not, run:

``` > pip install -r requirements.txt   ```

## Running the RAG
1. Open Docker Desktop. 
2. Run `docker compose up -d` to spin up the containers in detatched mode. This will start a redis-stack container, a weaviate container, and a chromadb container. 
3. Run `ingest.py` to index all the files in the `data` directory. These may include .pdf files, .py files, or .ipynb files. Select the embedding you wish to use, with the default being nomic. 

If you wish to run the RAG model in chroma or weaviate, use the corresponding `ingest` and `search` file. 

If the requested LLM or embedding model is not installed, the program will install it from ollama automatically. 

4. Run `search.py`. The UI will show in your console. Be sure to select the same embedding you used prior. 

5. Every query will generate a row in `data_collection.csv` that tracks variables including prompt and response length, LLM and embedding model used, system RAM, and response time. 

Note that there is not persistance. Closing and re-opening the container will require you to rerun `ingest.py`.
