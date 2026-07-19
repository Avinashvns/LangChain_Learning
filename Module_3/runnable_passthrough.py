from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough()

result = chain.invoke("Hello")
result_num = chain.invoke(30)

data = {
    "name" : "Avinash",
    "age" : 32
}
result_dict = chain.invoke(data)

print(result)
print(result_num)
print(result_dict)

#  real Rag Example

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ======================================
# Embedding Model
# ======================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ======================================
# Load Existing FAISS
# ======================================

vector_db = FAISS.load_local(
    "Module_2/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Vector Database Loaded Successfully")

# ======================================
# Retriever
# ======================================

retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# ======================================
# LLM
# ======================================

llm = ChatOllama(
    model="llama3.2"
)

# ======================================
# Prompt
# ======================================

prompt = ChatPromptTemplate.from_template(
"""
You are an AI Assistant.

Answer ONLY from the given context.

If the answer is not available in the context,
say:

I don't know.

Context:
{context}

Question:
{question}
"""
)

# ======================================
# Documents -> String
# ======================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# ======================================
# LCEL Chain
# ======================================

chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ======================================
# Chat
# ======================================

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(question)

    print("\nAI :")
    print(answer)