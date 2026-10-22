import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
embeddings_mode = "all-MiniLM-L6-v2"





def vector_loader(index_path,chunk_path):
    index = faiss.read_index(index_path)

    with open(chunk_path, "r",encoding="utf-8") as file:
        chunks = json.load(file)

    return index, chunks 


def reviewer(querry,index_path,chunk_path,k=4):
    index , chunks = vector_loader(index_path,chunk_path)
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )
    query_vector = model.encode_query(querry)
    query_vector = np.array([query_vector]).astype("float32")
    distances, indices = index.search(
        query_vector,
        k
    )

    results = []

    for distance, index_id in zip(
        distances[0],
        indices[0]
    ):

        results.append({
            "text": chunks[index_id]["text"],
            "metadata": chunks[index_id]["metadata"],
            "score": float(distance)
        })

    return results

if __name__ == "__main__" :
    result = reviewer("what  priyansh skilles","vector_store/index.faiss",r"vector_store\chunks.json",3)
    for r in result:
        print(r)