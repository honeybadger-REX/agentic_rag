import json
import ollama

from agent.tools import retrieve


MODEL = "qwen2.5:1.5b"


def decide(question: str, context: str) -> dict:

    context_text = context

    prompt = f"""
You are the controller of an Agentic RAG system.

Your job is to decide what action should happen next.

USER QUESTION:
{question}

RETRIEVED INFORMATION:
{context_text if context_text else "No information retrieved yet."}

AVAILABLE ACTIONS:

1. retrieve
Use this when you need more information.

2. answer
Use this only when you have enough information to answer
the user's question accurately.

If you choose retrieve, simply return the action.
Do not create a retrieval query.

Return ONLY JSON.

For retrieval:

{{
    "action": "retrieve",
}}
For evalutating retriev data:
{{
    "action":"evaluate"
}}

For evaluating answer :
{{
   "action" : "answer_evaluate
}}

For answering:

{{
    "action": "answer"
}}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "context":context
            }
        ],
        options={
            "temperature": 0
        }
    )

    text = response["message"]["content"].strip()

    print("\nRAW MODEL OUTPUT:")
    print(text)

    return json.loads(text)