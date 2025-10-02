import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from backend.core.embed import make_embeddings

load_dotenv()

DATA_DIR = os.getenv("RAG_DATA_DIR", "./data/rag")
INDEX_DIR = os.getenv("RAG_INDEX_DIR", "./storage/faiss")


def main(rebuild=False):
    loader = DirectoryLoader(DATA_DIR, glob="**/*", silent_errors=True)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    if rebuild and os.path.isdir(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    vs = FAISS.from_documents(chunks, make_embeddings())
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"[OK] Index FAISS → {INDEX_DIR} | chunks={len(chunks)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--rebuild", action="store_true")
    args = p.parse_args()
    main(rebuild=args.rebuild)
