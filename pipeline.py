import src.ingest as ir
import src.search as sr
import src.ingest_chroma as ic
import src.search_chroma as sc
# import src.ingest_weaviate as iw
# import src.search_weaviate as sw

llm_models = ['mistral:latest', 'llama3.2:latest']
embedding_models = ['nomic', 'minilm', 'mxbai']

def pipeline(prompts, chunk_sizes):
    for size in chunk_sizes:
        for model in embedding_models:
            # call redis ingest
            ir.main(model, chunk_size=size)
            # call chroma ingest
            ic.main(model, chunk_size=size)
            # call weaviate ingest
            # iw.main(model)
            for p in prompts:
                for llm in llm_models: 
                    # call redis search
                    sr.interactive_search(model, llm, p)
                    # call chroma search
                    sc.interactive_search(model, llm, p)
                    # call weaviate search
                    # sw.interactive_search(model, llm, p)


def main():
    prompts_list = [
        "What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?", 
        "When are linked lists faster than contiguously-allocated lists?",
        "What does the $nin operator mean in a Mongo query?"
        ]
    chunks_list = [70, 100, 300]
    pipeline(prompts_list, chunks_list)

if __name__ == "__main__":
    main()
