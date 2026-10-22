from sentence_transformers import CrossEncoder


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, results, top_k=4):

    pairs = [
        (query, result["text"])
        for result in results
    ]

    scores = reranker.predict(pairs)

    reranked = []

    for result, score in zip(results, scores):

        reranked.append({
            "text": result["text"],
            "page": result["metadata"],
            "score": float(score)
        })

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return reranked[:top_k]