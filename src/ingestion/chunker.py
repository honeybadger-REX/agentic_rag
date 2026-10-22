from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.loader import loader 

def chunking(document):

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    add_start_index = True,
    separators=["\n\n", "\n", ". ", " ", ""]
)
    chunks = splitter.split_documents(document)

    return {
        "chunk" : chunks,
        "lenght":len(chunks)

    }

if __name__ == "__main__":
    document = loader(r"C:\Users\HP\Downloads\Priyansh_Patel_AI_ML_Resume_New.pdf")
    an = chunking(document["document"])
    print(an["chunk"],an["lenght"])


# print(len(chunks))
# print(chunks[0].page_content)
# print(chunks[0].metadata)