from langchain_openai import ChatOpenAI
from src.generation.context_builder import build_context
from src.retrieval.retriever import reviewer



from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)



def generate_answer(query, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
          do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer



if __name__ == "__main__" :
    result = reviewer("what is priyansh skilles","vector_store\indexgujarat_police_camera_streaming_setup_guide.faiss","vector_store\chunksgujarat_police_camera_streaming_setup_guide.json",3)
    context =  build_context(result)
    r = generate_answer("what is priyansh skilles",context)
    print(r)

    