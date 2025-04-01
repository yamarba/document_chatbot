# NOTE: deprecated libraries
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.document_loaders import PyPDFLoader

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PDF_PATH = "./Rich-Dad-Poor-Dad.pdf"

# create loader
loader = PyPDFLoader(PDF_PATH)
# split document
pages = loader.load_and_split()

#print(len(pages))
# embedding function
embedding_func = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# create vector store
vectordb = Chroma.from_documents(
    documents=pages,
    embedding=embedding_func,
    persist_directory=f"./vector_db",
    collection_name="rich_dad_poor_dad")

# make vector store persistant
vectordb.persist()