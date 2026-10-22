import re
import math
import ollama


def evaluate_retrieval(results, threshold=0.0):

    if not results:
        return "bad"

    best_score = results[0]["score"]

    if best_score >= threshold:
        return "good"

    return "bad"

def evaluate_answer(answer: str, context: str, model="qwen2.5:1.5b") -> str:
    prompt = f"""Does the context support this answer? Reply YES or NO only.

Context: {context}
Answer: {answer}"""
    resp = ollama.chat(model=model, messages=[{"role":"user","content":prompt}],
                       options={"temperature": 0})
    if resp["message"]["content"].strip().upper().startswith("YES"):
        return "good"
    else:
        return"bad"

def evaluate_answer1(answer, context):
    """
    Check whether the generated answer is supported
    by the retrieved context.
    """

    if not answer or not answer.strip():
        return "bad"

    if not context or not context.strip():
        return "bad"

    # Split answer into sentences
    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        answer.strip()
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    context_words = set(
        re.findall(
            r"\b\w+\b",
            context.lower()
        )
    )

    good_count = 0

    for sent in sentences:

        sentence_words = set(
            re.findall(
                r"\b\w+\b",
                sent.lower()
            )
        )

        overlap = sentence_words.intersection(
            context_words
        )

        if len(overlap) >= 5:
            good_count += 1

    required_count = math.floor(
        len(sentences) * 0.6
    )

    if good_count >= required_count:
        return "good"

    return "bad"


