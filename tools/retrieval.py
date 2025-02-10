from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from models.llm_model import load_llm
from models.vector_store import load_vector_store

def qa_chat_prompt():
    qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. Use the provided context. If you don't know the answer, just say that you don't know. {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return qa_prompt

question_answer_chain = create_stuff_documents_chain(load_llm(), qa_chat_prompt())

vectorstore = load_vector_store("data/target_audience.csv")


# 5. Initialize retrieval chain
history_aware_retriever = vectorstore.as_retriever()
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) 
