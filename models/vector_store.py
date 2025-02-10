from langchain_core.vectorstores import InMemoryVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

def load_vector_store(csv_path):
    """Load and store document embeddings."""
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, OpenAIEmbeddings())
    vectorstore.save_local("data/faiss_index")  # Saves for reuse
    vectorstore = FAISS.load_local("data/faiss_index", OpenAIEmbeddings())

    return vectorstore

