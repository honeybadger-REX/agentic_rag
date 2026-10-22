from langchain_community.document_loaders import PyPDFLoader


def loader(FILE_PATH):
    loader = PyPDFLoader(FILE_PATH)
    document = loader.load()
    return {
        "document" :document,
        "lenght": len(document)
    }

if __name__ == "__main__":
    returns = loader(input(str).strip())
    print(returns["document"])
