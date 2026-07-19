from langchain_core.runnables import RunnableParallel , RunnableLambda

def uppercase(text):
    return text.upper()

def lowercase(text):
    return text.lower()

def length(text):
    return len(text)

chain = RunnableParallel(
    upper = RunnableLambda(uppercase),
    lower = RunnableLambda(lowercase),
    count = RunnableLambda(length)
)

result = chain.invoke("LangChain")
print(result)

# Real RAG Example

from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================================
# Embedding Model
# ============================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ============================================
# Load Existing FAISS Database
# ============================================

vector_db = FAISS.load_local(
    "Module_2/vector_store",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Vector Database Loaded Successfully")

# ============================================
# Retriever
# ============================================

retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)

# ============================================
# LLM
# ============================================

llm = ChatOllama(
    model="llama3.2"
)

# ============================================
# Prompt
# ============================================

prompt = ChatPromptTemplate.from_template(
"""
You are an AI Assistant.

Answer ONLY from the given context.

If the answer is not available in the context,
reply with:

I don't know.

Context:
{context}

Question:
{question}
"""
)

# ============================================
# Convert Documents → String
# ============================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# ============================================
# Runnable Parallel
# ============================================

parallel = RunnableParallel(

    context = retriever | RunnableLambda(format_docs),

    question = RunnablePassthrough()

)

# ============================================
# LCEL Chain
# ============================================

chain = (

    parallel

    | prompt

    | llm

    | StrOutputParser()

)

# ============================================
# Chat Loop
# ============================================

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    answer = chain.invoke(question)

    print("\nAI :")
    print(answer)