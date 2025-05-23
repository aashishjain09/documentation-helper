from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OllamaEmbeddings(model="llama3")


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/python.langchain.com/docs")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    # documents = [
    #     doc.metadata.update(
    #         {"source": doc.metadata["source"].replace("langchain-docs", "https:/")}
    #     )
    #     for doc in documents
    # ]

    print(f"Going to add {len(documents)} to Pinecone")

    PineconeVectorStore.from_documents(
        documents, embeddings, index_name="langchain-doc-index"
    )

    print("**** Loading to vectorstore done ****")


if __name__ == "__main__":
    ingest_docs()
