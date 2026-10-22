import json
import ollama


MODEL = "qwen2.5:1.5b"


def decide(question: str, context: str) -> dict:

    # -----------------------------------------
    # AGENT PROMPT
    # -----------------------------------------

    prompt = f"""
You are the controller of an Agentic RAG system.

Your job is to decide what action should happen next.

You do NOT answer the user's question yourself.

You only choose ONE action from the available actions.

USER QUESTION:
{question}

CURRENT RETRIEVED INFORMATION:
{context if context else "No information retrieved yet."}


AVAILABLE ACTIONS:

1. retrieve

Use this when no useful information has been retrieved yet.

Return:

{{"action": "retrieve"}}


2. retrieval_evaluate

Use this after information has been retrieved.

Return:

{{"action": "retrieval_evaluate"}}


3. corrective_retrieve

Use this when the previous retrieval was not good enough.

Return:

{{"action": "corrective_retrieve"}}


4. answer

Use this when the retrieved information is sufficient to answer
the user's question.

Return:

{{"action": "answer"}}


5. answer_evaluate

Use this after an answer has been generated.

Return:

{{"action": "answer_evaluate"}}


6. finish

Use this when the answer has already been evaluated as good.

Return:

{{"action": "finish"}}


IMPORTANT:

Return ONLY valid JSON.

Do not write explanations.

Choose exactly ONE action.

"""


    # -----------------------------------------
    # CALL OLLAMA
    # -----------------------------------------

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": 0,
        },
    )


    # -----------------------------------------
    # READ MODEL RESPONSE
    # -----------------------------------------

    text = response["message"]["content"].strip()

    print("\nRAW MODEL OUTPUT:")
    print(text)


    # -----------------------------------------
    # PARSE JSON
    # -----------------------------------------

    try:

        decision = json.loads(text)

    except json.JSONDecodeError:

        print("\nAgent returned invalid JSON.")

        # Safe fallback
        decision = {
            "action": "retrieve"
        }


    # -----------------------------------------
    # VALIDATE ACTION
    # -----------------------------------------

    valid_actions = {
        "retrieve",
        "retrieval_evaluate",
        "corrective_retrieve",
        "answer",
        "answer_evaluate",
        "finish",
    }

    action = decision.get("action")

    if action not in valid_actions:

        print("\nInvalid action from agent:", action)

        decision = {
            "action": "retrieve"
        }


    return decision