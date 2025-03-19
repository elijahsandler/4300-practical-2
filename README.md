# DS4300 - Practical 2
Ensure all libraries in `requirements.txt` are installed. If they are not, run:

``` > pip install -r requirements.txt   ```

## Running the RAG
1. Open Docker Desktop. Ensure the `redis/redis-stack:latest` image is pulled. 
2. Run `docker compose up -d` to spin up the container in detatched mode. 
3. Run `ingest.py` to index all the files in the `data` directory. These may include pdfs, .py files, or .ipynb files. 
4. Run `search.py`. The UI will show in your console. 

Note that there is not persistance in the redis-stack container. Closing and re-opening the container will require you to rerun `ingest.py`.