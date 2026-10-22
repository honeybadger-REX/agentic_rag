from pathlib import Path
from src.retrieval.retriever import reviewer
from src.retrieval.reranker import rerank
from src.generation.context_builder import build_context
from src.generation.genrater import generate_answer
from src.evaluator.corrective_reranking import corrective_reranking
from src.evaluator.evaluate_retrieval import evaluate_retrieval

KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "knowledge.txt"

knowledge = KNOWLEDGE_PATH.read_text(encoding="utf-8")




# RETRIVER
# ---------------------------------------------


def tools(query:str,tool:dict):
    if tool["name"] == "retriver" :
        result = reviewer(query,tool["index_pt"],tool["chunk_pt"],tool["retrive_top"])
        reranked = rerank(query,result,tool["rerank_top"])
        context = build_context(reranked)
        return {"context" : context,"highest score":reranked["score"][0]}





















# EXAMPLE

# def retrieve(query: str) -> list[str]:
#     """
#     Very simple keyword-based retrieval.

#     This is temporary.
#     Later we will replace this with embeddings + vector search.
#     """

#     chunks = knowledge.split("\n\n")

#     query_words = query.lower().split()

#     results = []

#     for chunk in chunks:

#         score = 0

#         for word in query_words:
#             if word in chunk.lower():
#                 score += 1

#         if score > 0:
#             results.append((score, chunk))

#     results.sort(key=lambda x: x[0], reverse=True)

#     return [chunk for score, chunk in results[:3]]