from langchain_core.runnables import RunnableBranch, RunnableLambda

python_chain = RunnableLambda(
    lambda x: "python Expert"
)

general_chain = RunnableLambda(
    lambda x: "General Expert"
)

branch = RunnableBranch(
    (
        lambda x: "python" in x.lower(),
        python_chain
    ),
    general_chain
)

print(branch.invoke("What is Python?"))
print(branch.invoke("Tell me a joke"))

# Real Example RAG

from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2"
)

python_prompt = ChatPromptTemplate.from_template(
"""
You are a Python Expert.

Answer:

{question}
"""
)

general_prompt = ChatPromptTemplate.from_template(
"""
You are a Helpful AI Assistant.

Answer:

{question}
"""
)

python_chain = (
    python_prompt
    | llm
    | StrOutputParser()
)

general_chain = (
    general_prompt
    | llm
    | StrOutputParser()
)

branch = RunnableBranch(

    (
        lambda x: "python" in x["question"].lower(),
        python_chain
    ),

    general_chain
)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    answer = branch.invoke(
        {
            "question": question
        }
    )

    print("\nAI :")
    print(answer)