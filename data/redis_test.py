import redis
from tqdm import tqdm
import json
import os

def index_files(path: str):
    """
    Indexes all JSON files in the given folder into the given data structure.

    Args:
        path (str): The path to the folder containing JSON files.
    """

    # Walk through the directory and index all JSON files
    for root, _, files in os.walk(path):
        for file in tqdm(files):
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # create a pipeline for batch execution
                pipe = r.pipeline()

                # Open and load JSON data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    words = data['preprocessed_text']

                    # Insert unique words into the pipeline
                    for word in set(words):
                        pipe.sadd(word, data['title'])

                # Execute the pipeline
                res = pipe.execute()

                # Check if all operations were successful
                assert all(res), f"failed to index {file_path}"

def main():
    # establish redis connection
    r = redis.Redis(host='localhost', port=6379, db=0)
    assert r.ping(), 'redis connection not established'
    
    index_files('USFinancialNewsArticles-medium')