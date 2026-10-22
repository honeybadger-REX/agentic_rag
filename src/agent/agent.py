import json
import ollama

MODEL = "qwen2.5:1.5b"


def decide(
    question: str,
    context: str,
    answer: str = "",
    last_action: str = ""
) -> dict:

    prompt = f"""
You are the controller of an Agentic RAG system.

Your job is ONLY to decide the next action.
You must NOT answer the user's question.

USER QUESTION:
{question}

RETRIEVED INFORMATION:
{context if context else "No information retrieved yet."}

CURRENT ANSWER:
{answer if answer else "No answer generated yet."}

LAST ACTION:
{last_action if last_action else "None"}

AVAILABLE ACTIONS:

1. retrieve
Use this when no information has been retrieved yet.

Return:
{{"action": "retrieve"}}


2. retrieval_evaluate
Use this after retrieval has happened.

Return:
{{"action": "retrieval_evaluate"}}


3. corrective_retrieve
Use this when retrieval evaluation says the retrieved information is bad.

Return:
{{"action": "corrective_retrieve"}}


4. answer
Use this when retrieval is good and there is no answer yet.

Return:
{{"action": "answer"}}


5. answer_evaluate
Use this after an answer has been generated.

Return:
{{"action": "answer_evaluate"}}


6. finish
Use this when the answer evaluation says the answer is good.

Return:
{{"action": "finish"}}


IMPORTANT WORKFLOW RULES:

- If there is no retrieved information, choose "retrieve".
- If the last action was "retrieve", choose "retrieval_evaluate".
- If retrieval evaluation was bad, choose "corrective_retrieve".
- If retrieval is good and there is no answer, choose "answer".
- If the last action was "answer", choose "answer_evaluate".
- If answer evaluation is good, choose "finish".
- Never choose "answer" repeatedly.
- Never answer the user yourself.
- Return ONLY valid JSON.
- Choose exactly ONE action.

JSON FORMAT:
{{"action": "retrieve"}}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    text = response["message"]["content"].strip()

    print("\nRAW MODEL OUTPUT:")
    print(text)

    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        print("\nInvalid JSON from agent. Defaulting to retrieve.")
        return {"action": "retrieve"}

    valid_actions = {
        "retrieve",
        "retrieval_evaluate",
        "corrective_retrieve",
        "answer",
        "answer_evaluate",
        "finish",
    }

    if decision.get("action") not in valid_actions:
        print("\nInvalid action:", decision.get("action"))
        return {"action": "retrieve"}

    return decision