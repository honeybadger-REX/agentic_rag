from src.evaluator.coracttive_retrieval import corrective_retrieval
from src.evaluator.evaluate_retrieval import evaluate_retrieval
from src.retrieval.reranker import rerank


def corrective_reranking(query,index_path,chunk_path,k=5):

    results = corrective_retrieval(query,index_path,chunk_path,k=5)

    reranked = rerank(
        query,
        results,
        top_k=4
    )
    if evaluate_retrieval(reranked,0.3) == "good":
        return reranked
    else:
        return "bad"
    
from src.evaluator.coracttive_retrieval import corrective_retrieval
from src.retrieval.reranker import rerank

# TEST
def corrective_rerankingt(
    query,
    index_path,
    chunk_path,
    k=5
):

    results = corrective_retrieval(
        query,
        index_path,
        chunk_path,
        k=k
    )

    reranked = rerank(
        query,
        results,
        top_k=4
    )

    return reranked