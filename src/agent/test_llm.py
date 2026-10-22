import ollama

response = ollama.chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one sentence."
        }
    ]
)

print(response["message"]["content"])