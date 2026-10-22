import ollama


MODEL = "qwen2.5:1.5b"


def generate_answer(query, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information,
say that you do not have enough information.

Context:
{context}

Question:
{query}

Answer:
"""

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

    return response["message"]["content"].strip()