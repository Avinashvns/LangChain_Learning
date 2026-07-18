from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------
# Load Embedding Model
# ---------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ---------------------------------
# Load Existing FAISS Database
# ---------------------------------

vector_db = FAISS.load_local(
    folder_path="Module_2/vector_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS Database Loaded Successfully")

# ---------------------------------
# Create Retriever
# ---------------------------------

retriever = vector_db.as_retriever(
    search_kwargs={
        "k":3
    }
)

# ---------------------------------
# Load LLM
# ---------------------------------

llm = ChatOllama(
    model="llama3.2"
)

# ---------------------------------
# Prompt
# ---------------------------------

prompt = ChatPromptTemplate.from_template(
"""
Answer the question only from the given context.

Context:
{context}

Question:
{question}
"""
)

# ---------------------------------
# Chat Loop
# ---------------------------------

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    # Retrieve Documents
    docs = retriever.invoke(question)

    # Combine Context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Create Chain
    chain = prompt | llm | StrOutputParser()

    # Generate Answer
    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    print("\nAI :")
    print(answer)