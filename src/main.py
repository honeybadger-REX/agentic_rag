import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.agent.agent import decide
from src.agent.tools import tools



MAX_STEPS = 5



def run_agent(question: str):

    # Agent state
    context = ""
    results = []
    answer = ""

    for step in range(MAX_STEPS):

        print(f"\n========== STEP {step + 1} ==========")

        # -----------------------------------------
        # AGENT DECIDES WHAT TO DO
        # -----------------------------------------

        decision = decide(question, context)

        print("\nDECISION:")
        print(decision)

        action = decision.get("action")

        # -----------------------------------------
        # RETRIEVE
        # -----------------------------------------

        if action == "retrieve":

            tool = {
                "name": "retrieve",
                "index_pt": r"src\vector_store\IELTS_Grammar_Quick_Revision_Priyansh_index.faiss",
                "chunk_pt": r"src\vector_store\IELTS_Grammar_Quick_Revision_Priyansh_chunks.json",
                "rerank_top": 5,
                "retrive_top": 8,
            }

            print("\nTOOL CALL:")
            print(f"retrieve('{question}')")

            documents = tools(question, tool)

            if not documents["results"]:
                print("\nNo documents found.")
                break

            results = documents["results"]
            context = documents["context"]

            print("\nRETRIEVED:")

            for document in results:
                print("-", document["text"])

        # -----------------------------------------
        # RETRIEVAL EVALUATION
        # -----------------------------------------

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

            # If retrieval is bad, clear the current
            # context so the agent can decide to
            # perform corrective retrieval.
            if evaluation["status"] == "bad":
                context = ""

        # -----------------------------------------
        # CORRECTIVE RETRIEVAL
        # -----------------------------------------

        elif action == "corrective_retrieve":

            tool = {
                "name": "corrective_retrieve",
                "index_pt":r"src\vector_store\IELTS_Grammar_Quick_Revision_Priyansh_index.faiss" ,
                "chunk_pt": r"src\vector_store\IELTS_Grammar_Quick_Revision_Priyansh_chunks.json",
                "retrive_top": 5,
            }

            print("\nTOOL CALL:")
            print(f"corrective_retrieve('{question}')")

            documents = tools(question, tool)

            if documents["status"] == "bad":
                print("\nCorrective retrieval failed.")
                break

            results = documents["results"]
            context = documents["context"]

            print("\nCORRECTIVE RETRIEVAL:")

            for document in results:
                print("-", document["text"])

        # -----------------------------------------
        # GENERATE ANSWER
        # -----------------------------------------

        elif action == "answer":

            if not context:
                print("\nNo context available for answering.")
                break

            tool = {
                "name": "answer",
                "context": context,
            }

            print("\nTOOL CALL:")
            print("answer()")

            answer_result = tools(question, tool)

            answer = answer_result["answer"]

            print("\nGENERATED ANSWER:")
            print(answer)

        # -----------------------------------------
        # ANSWER EVALUATION
        # -----------------------------------------

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

                print("\n========== FINAL ANSWER ==========")
                print(answer)

                return answer

            else:

                # Bad answer.
                # Let the agent decide what to do next.
                answer = ""

        # -----------------------------------------
        # FINISH
        # -----------------------------------------

        elif action == "finish":

            print("\n========== FINAL ANSWER ==========")
            print(answer)

            return answer

        # -----------------------------------------
        # UNKNOWN ACTION
        # -----------------------------------------

        else:

            print("\nUnknown action:", action)
            break

    print("\nAgent stopped: maximum steps reached.")

    if answer:
        print("\nLast generated answer:")
        print(answer)

    return answer


if __name__ == "__main__":

    question = input("\nAsk a question: ")

    run_agent(question)