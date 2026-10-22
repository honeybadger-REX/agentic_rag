from pathlib import Path

from src.retrieval.retriever import reviewer
from src.retrieval.reranker import rerank
from src.generation.context_builder import build_context
from src.generation.genrater import generate_answer
from src.evaluator.corrective_reranking import corrective_reranking
from src.evaluator.evaluate_retrieval import (
    evaluate_retrieval,
    evaluate_answer,
)


# KNOWLEDGE_PATH = Path(__file__).parent / "knowledge" / "knowledge.txt"

# knowledge = KNOWLEDGE_PATH.read_text(encoding="utf-8")


def tools(query: str, tool: dict):

    # -------------------------------------------------
    # RETRIEVE
    # -------------------------------------------------

    if tool["name"] == "retrieve":

        result = reviewer(
            query,
            tool["index_pt"],
            tool["chunk_pt"],
            tool.get("retrive_top", 8),
        )

        if not result:
            return {
                "context": "",
                "results": [],
                "highest_score": None,
            }

        reranked = rerank(
            query,
            result,
            tool.get("rerank_top", 5),
        )

        if not reranked:
            return {
                "context": "",
                "results": [],
                "highest_score": None,
            }

        context = build_context(reranked)

        return {
            "context": context,
            "results": reranked,
            "highest_score": reranked[0]["score"],
        }


    # -------------------------------------------------
    # RETRIEVAL EVALUATION
    # -------------------------------------------------

    elif tool["name"] == "retrieval_evaluate":

        results = tool.get("result", [])

        status = evaluate_retrieval(
            results,
            tool.get("threshold", -0.08),
        )

        if status == "bad":
            return {
                "status": "bad",
                "message": "retrieval is not good",
            }

        return {
            "status": "good",
            "message": "retrieval is good for answering",
        }


    # -------------------------------------------------
    # CORRECTIVE RETRIEVAL
    # -------------------------------------------------

    elif tool["name"] == "corrective_retrieve":

        result = corrective_reranking(
            query,
            tool["index_pt"],
            tool["chunk_pt"],
            tool.get("retrive_top", 5),
        )

        if result == "bad" or not result:
            return {
                "status": "bad",
                "context": "",
                "results": [],
            }

        context = build_context(result)

        return {
            "status": "good",
            "context": context,
            "results": result,
            "highest_score": result[0]["score"],
        }


    # -------------------------------------------------
    # GENERATE ANSWER
    # -------------------------------------------------

    elif tool["name"] == "answer":

        context = tool.get("context", "")

        if not context:
            return {
                "answer": "",
                "status": "bad",
                "message": "No context available for answering.",
            }

        answer = generate_answer(
            query,
            context,
        )

        return {
            "answer": answer,
            "status": "generated",
        }


    # -------------------------------------------------
    # ANSWER EVALUATION
    # -------------------------------------------------

    elif tool["name"] == "answer_evaluate":

        answer = tool.get("answer", "")
        context = tool.get("context", "")

        status = evaluate_answer(
            answer,
            context,
        )

        if status == "bad":
            return {
                "status": "bad",
                "message": "answer is not good enough",
            }

        return {
            "status": "good",
            "message": "answer is good to return",
        }


    # -------------------------------------------------
    # UNKNOWN TOOL
    # -------------------------------------------------

    else:

        return {
            "status": "error",
            "message": f"Tool '{tool.get('name')}' is not available.",
        }
















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