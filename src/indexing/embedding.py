from src.ingestion.loader import loader
from src.ingestion.chunker import chunking
import ollama


EMBEDDING_MODEL = "nomic-embed-text"


def embedding(chunks):

    # Convert LangChain Document objects to text
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # Generate embeddings using Ollama
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    embeddings = response["embeddings"]

    return {
        "embedding": embeddings,
        "shape": (len(embeddings), len(embeddings[0]))
    }


if __name__ == "__main__":

    document = loader(
        r"C:\Users\HP\Downloads\Priyansh_Patel_AI_ML_Resume_New.pdf"
    )

    an = chunking(
        document["document"]
    )

    em = embedding(
        chunks=an["chunk"]
    )

    print("Embedding shape:", em["shape"])
    print("Number of embeddings:", len(em["embedding"]))