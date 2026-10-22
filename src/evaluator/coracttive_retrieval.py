from src.retrieval.retriever import reviewer as retrieve

def corrective_retrieval(query,index_path,chunk_path,k=5):

    corrective_queries = [
        query,
        f"{query} policy",
        f"{query} details",
        f"{query} terms"
    ]

    all_results = []

    for q in corrective_queries:

        results = retrieve(
            q,
            index_path,
            chunk_path,
            k
            
        )

        all_results.extend(results)
        print(all_results)

    return all_results