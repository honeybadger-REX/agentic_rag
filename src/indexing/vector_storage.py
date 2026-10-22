import faiss
import numpy as np
import os
import json

from src.ingestion.loader import loader
from src.ingestion.chunker import chunking
from src.indexing.embedding import embedding


def indexing(embeddings):

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    # Create FAISS index using L2 distance
    index = faiss.IndexFlatL2(
        dimension
    )

    # Add vectors to index
    index.add(embeddings)

    print("Embedding shape:", embeddings.shape)
    print("Vector dimension:", dimension)
    print("Vectors indexed:", index.ntotal)

    return index


def save_index(index, chunks,index_path,chunk_path):

    os.makedirs(
        "vector_store",
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(
        index,
        index_path
    )

    # Save original chunks
    chunk_data = [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]

    with open(chunk_path,"w",encoding="utf-8") as file:

        json.dump(
            chunk_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    return {"message":"Index saved successfully."}


if __name__ == "__main__":

    document = loader(
        r"C:\Users\HP\Downloads\Priyansh_Patel_AI_ML_Resume_New.pdf"
    )

    # Chunking
    an = chunking(
        document["document"]
    )

    # Embedding
    em = embedding(
        chunks=an["chunk"]
    )

    # FAISS indexing
    index = indexing(
        em["embedding"]
    )

    # Save index + chunks
    save_index(
        index,
        an["chunk"]
    )