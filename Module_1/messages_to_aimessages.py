from langchain_ollama import ChatOllama
from langchain_core.messages import (SystemMessage,HumanMessage,AIMessage)

llm = ChatOllama(model="llama3.2")

messages = [
    SystemMessage(content="You are a Python teacher."),
    HumanMessage(content="What is Python?"),
    AIMessage(content="Python is a programming language."),
    HumanMessage(content="Who created it?")
]

response = llm.invoke(messages)
print(response.content)