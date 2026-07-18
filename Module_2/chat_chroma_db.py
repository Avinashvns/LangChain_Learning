from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------------
# Embedding Model
# -------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -------------------------
# Load Existing Chroma
# -------------------------

vector_db = Chroma(
    collection_name="python_docs",
    embedding_function=embeddings,
    persist_directory="Module_2/chroma_db"
)

print("Chroma Database Loaded Successfully")

# -------------------------
# Retriever
# -------------------------

retriever = vector_db.as_retriever(
    search_kwargs={
        "k":3
    }
)

# -------------------------
# LLM
# -------------------------

llm = ChatOllama(
    model="llama3.2"
)

# -------------------------
# Prompt
# -------------------------

prompt = ChatPromptTemplate.from_template(
"""
Answer only from the given context.

Context:
{context}

Question:
{question}
"""
)

# -------------------------
# Create Chain
# -------------------------

chain = prompt | llm | StrOutputParser()

# -------------------------
# Chat Loop
# -------------------------

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    print("\nAI :")
    print(answer)