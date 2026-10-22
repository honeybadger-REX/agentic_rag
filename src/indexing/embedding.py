from sentence_transformers import SentenceTransformer
from src.ingestion.loader import loader
from src.ingestion.chunker import chunking


def embedding(chunks):

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Convert LangChain Document objects to text
    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    # Convert text to vectors
    embeddings = model.encode(texts)

    return {
        "embedding": embeddings,
        "shape": embeddings.shape
    }


if __name__ == "__main__":

    document = loader(
        r"C:\Users\HP\Downloads\Priyansh_Patel_AI_ML_Resume_New.pdf"
    )

    an = chunking(
        document["document"]
    )

    print(an["chunk"])
 

    em = embedding(
        chunks=an["chunk"]
    )

    print("Embedding shape:", em["shape"])
    print(em["embedding"])