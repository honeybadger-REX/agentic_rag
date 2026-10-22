from agent import decide
from agent.tools import retrieve
from src.document_embeding import loader


MAX_STEPS = 5

load = loader()
def run_agent(question: str):

    context = []

    for step in range(MAX_STEPS):

        print(f"\n========== STEP {step + 1} ==========")

        decision = decide(question, context)

        print("\nDECISION:")
        print(decision)

        action = decision["action"]  

        if action == "retrieve":
            tool = {
                "name":"retriver",
                "index_pt": load.index_pt,
                "chunk_pt": load.chunks_pt,
                "rerank_top":5,
                "retrive_top":8

            }

            query = question 

            print("\nTOOL CALL:")
            print(f"retrieve('{query}')")

            documents = tools(query,tool)

            if not documents:
                print("\nNo documents found.")
                break

            print("\nRETRIEVED:")
            for document in documents["context"]:
                print("-", document)

            context.extend(documents)
        elif action == "reavulae":
            pass

        elif action == "answer":

            print("\nAgent decided it has enough information.")
            

            print("\nFINAL CONTEXT:")
            for document in context:
                print("-", document)

            return

        else:

            print("Unknown action:", action)
            break


if __name__ == "__main__":

    question = input("\nAsk a question: ")

    run_agent(question)