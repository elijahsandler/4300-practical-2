import src.ingest as ir
import src.search as sr
import src.ingest_chroma as ic
import src.search_chroma as sc
import src.ingest_weaviate as iw
import src.search_weaviate as sw

llm_models = ['', '']
embedding_models = ['', '', '']

def pipeline(prompts):
    for model in embedding_models:
        # call redis ingest
        ir.main()
        # call chroma ingest
        ic.main()
        # call weaviate ingest
        iw.main()
        for p in prompts:
            for llm in llm_models:
                # call redis search
                # call chroma search
                # call weaviate search
                pass


def main():
    pipeline()

if __name__ == "__main__":
    main()
