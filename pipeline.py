import src.ingest as ir
import src.search as sr
import src.ingest_chroma as ic
import src.search_chroma as sc
# import src.ingest_weaviate as iw
# import src.search_weaviate as sw

llm_models = ['mistral:latest', 'llama3.2:latest']
embedding_models = ['nomic', 'minilm', 'mxbai']

def pipeline(prompts):
    for model in embedding_models:
        # call redis ingest
        ir.main(model)
        # call chroma ingest
        # ic.main(model)
        # call weaviate ingest
        # iw.main(model)
        for p in prompts:
            for llm in llm_models:
                # call redis search
                sr.interactive_search(model, llm, p)
                # call chroma search
                # sc.interactive_search(model, p)
                # call weaviate search
                # sw.interactive_search(model, p)


def main():
    prompts_list = ["redis", "avl"]
    pipeline(prompts_list)

if __name__ == "__main__":
    main()
