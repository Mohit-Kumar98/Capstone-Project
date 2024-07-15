from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data"

def main():

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    print(chunks[0:2])


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


if __name__ == "__main__":
    main()


# embed documents
from langchain_openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings(model="text-embedding-3-large")

embeddings = embed_model.embed_documents([text])