#NOTE: deprecated libraries
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain.vectorstores import Chroma
#from langchain.prompts import PromptTemplate
#from langchain.chains import ConversationalRetrievalChain
#from langchain.chat_models import ChatOpenAI
#from langchain.memory import ConversationBufferMemory
#from langchain.document_loaders import PyPDFLoader
import os
from decouple import config
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
#from langchain_core.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

os.environ["TOKENIZERS_PARALLELISM"] = "false"

embedding_function = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# embedding_function = SentenceTransformerEmbeddings(
#     model_name="all-MiniLM-L6-v2"
# )

vector_db = Chroma(
    persist_directory="./vector_db",
    collection_name="rich_dad_poor_dad",
    embedding_function=embedding_function,
)


# create prompt
QA_prompt = PromptTemplate(
    template="""Use the following pieces of context to answer the user question.
chat_history: {chat_history}
Context: {text}
Question: {question}
Answer:""",
    input_variables=["text", "question", "chat_history"]
)

# create chat model
llm = ChatOpenAI(openai_api_key=config("OPENAI_API_KEY"), temperature=0)

# create memory
#memory = ConversationBufferMemory(return_messages=True)
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")


# memory = ConversationBufferMemory(
#     return_messages=True, memory_key="chat_history")

# create retriever chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    retriever=vector_db.as_retriever(
        search_kwargs={'fetch_k': 4, 'k': 3}, search_type='mmr'),
    chain_type="refine",
)

# question
#question = "What is the book about?"


def rag(question: str) -> str:
    # call QA chain
#    response = qa_chain({"question": question})
    response = qa_chain.invoke({"question": question})

    return response.get("answer")