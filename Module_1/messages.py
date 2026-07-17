from langchain_ollama import ChatOllama
from langchain_core.messages import ( HumanMessage, SystemMessage )

llm = ChatOllama(model="llama3.2")

messages = [
    SystemMessage(content="You are a helpful AI teacher."),
    HumanMessage(content="Explain Python.")
]

response = llm.invoke(messages)

print("Content :",response)