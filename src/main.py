import sys
from pathlib import Path

# ---------------------------------------------------------
# PROJECT ROOT
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

from src.agent.agent import decide
from src.agent.tools import tools


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

MAX_STEPS = 8

VECTOR_STORE = ROOT_DIR / "src" / "vector_store"

INDEX_PATH = (
    VECTOR_STORE
    / "IELTS_Grammar_Quick_Revision_Priyansh_index.faiss"
)

CHUNK_PATH = (
    VECTOR_STORE
    / "IELTS_Grammar_Quick_Revision_Priyansh_chunks.json"
)


# ---------------------------------------------------------
# CHECK VECTOR STORE
# ---------------------------------------------------------

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"FAISS index not found:\n{INDEX_PATH}"
    )

if not CHUNK_PATH.exists():
    raise FileNotFoundError(
        f"Chunk file not found:\n{CHUNK_PATH}"
    )


# ---------------------------------------------------------
# AGENT LOOP
# ---------------------------------------------------------

def run_agent(question: str):

    # Current agent state
    context = ""
    results = []
    answer = ""

    # Used by the controller to understand what happened
    # in the previous step.
    last_action = ""

    for step in range(1, MAX_STEPS + 1):

        print(f"\n{'=' * 50}")
        print(f"STEP {step}")
        print(f"{'=' * 50}")

        # -------------------------------------------------
        # AGENT DECIDES NEXT ACTION
        # -------------------------------------------------

        decision = decide(
            question=question,
            context=context,
            answer=answer,
            last_action=last_action,
        )

        print("\nDECISION:")
        print(decision)

        action = decision.get("action")

        # -------------------------------------------------
        # RETRIEVE
        # -------------------------------------------------

        if action == "retrieve":

            tool = {
                "name": "retrieve",
                "index_pt": str(INDEX_PATH),
                "chunk_pt": str(CHUNK_PATH),
                "rerank_top": 5,
                "retrive_top": 8,
            }

            print("\nTOOL CALL:")
            print(f"retrieve('{question}')")

            documents = tools(question, tool)

            if not documents.get("results"):

                print("\nNo documents found.")

                last_action = "retrieve"

                break

            results = documents["results"]
            context = documents["context"]

            print("\nRETRIEVED DOCUMENTS:")

            for document in results:
                print("-", document["text"])

            print(
                "\nHighest retrieval score:",
                documents.get("highest_score")
            )

        # -------------------------------------------------
        # RETRIEVAL EVALUATION
        # -------------------------------------------------

        elif action == "retrieval_evaluate":

            tool = {
                "name": "retrieval_evaluate",
                "result": results,
                "threshold": -0.08,
            }

            print("\nTOOL CALL:")
            print("retrieval_evaluate()")

            evaluation = tools(question, tool)

            print("\nRETRIEVAL EVALUATION:")
            print(evaluation)

            if evaluation["status"] == "bad":

                print(
                    "\nRetrieval is BAD."
                    "\nAgent should perform corrective retrieval."
                )

                context = ""

            else:

                print(
                    "\nRetrieval is GOOD."
                    "\nAgent can continue to answer."
                )

        # -------------------------------------------------
        # CORRECTIVE RETRIEVAL
        # -------------------------------------------------

        elif action == "corrective_retrieve":

            tool = {
                "name": "corrective_retrieve",
                "index_pt": str(INDEX_PATH),
                "chunk_pt": str(CHUNK_PATH),
                "retrive_top": 5,
            }

            print("\nTOOL CALL:")
            print(f"corrective_retrieve('{question}')")

            documents = tools(question, tool)

            if documents.get("status") == "bad":

                print("\nCorrective retrieval failed.")

                last_action = "corrective_retrieve"

                break

            results = documents["results"]
            context = documents["context"]

            print("\nCORRECTIVE RETRIEVAL:")

            for document in results:
                print("-", document["text"])

            print(
                "\nHighest retrieval score:",
                documents.get("highest_score")
            )

        # -------------------------------------------------
        # GENERATE ANSWER
        # -------------------------------------------------

        elif action == "answer":

            if not context:

                print(
                    "\nNo context available for answering."
                )

                last_action = "answer"

                continue

            tool = {
                "name": "answer",
                "context": context,
            }

            print("\nTOOL CALL:")
            print("answer()")

            answer_result = tools(question, tool)

            answer = answer_result.get("answer", "")

            print("\nGENERATED ANSWER:")
            print(answer)

            if not answer:

                print(
                    "\nAnswer generation failed."
                )

        # -------------------------------------------------
        # ANSWER EVALUATION
        # -------------------------------------------------

        elif action == "answer_evaluate":

            tool = {
                "name": "answer_evaluate",
                "answer": answer,
                "context": context,
            }

            print("\nTOOL CALL:")
            print("answer_evaluate()")

            evaluation = tools(question, tool)

            print("\nANSWER EVALUATION:")
            print(evaluation)

            if evaluation["status"] == "good":

                print(
                    "\nAnswer evaluation is GOOD."
                )

                print(
                    "\n"
                    + "=" * 50
                )
                print("FINAL ANSWER")
                print("=" * 50)

                print(answer)

                return answer

            else:

                print(
                    "\nAnswer evaluation is BAD."
                )

                # Remove bad answer.
                answer = ""

                print(
                    "Answer cleared."
                    "\nAgent will decide the next action."
                )

        # -------------------------------------------------
        # FINISH
        # -------------------------------------------------

        elif action == "finish":

            print(
                "\n"
                + "=" * 50
            )
            print("FINAL ANSWER")
            print("=" * 50)

            print(answer)

            return answer

        # -------------------------------------------------
        # UNKNOWN ACTION
        # -------------------------------------------------

        else:

            print(
                "\nUnknown action:",
                action
            )

            break

        # -------------------------------------------------
        # UPDATE STATE
        # -------------------------------------------------

        last_action = action

        print("\nCURRENT STATE:")
        print("last_action :", last_action)
        print("context     :", bool(context))
        print("answer      :", bool(answer))

    # -----------------------------------------------------
    # MAX STEPS REACHED
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 50
    )
    print("AGENT STOPPED")
    print("=" * 50)

    print(
        f"Maximum steps ({MAX_STEPS}) reached."
    )

    if answer:

        print(
            "\nLast generated answer:"
        )

        print(answer)

    return answer


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    question = input(
        "\nAsk a question: "
    ).strip()

    if not question:

        print(
            "\nQuestion cannot be empty."
        )

    else:

        run_agent(question)