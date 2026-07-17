from langchain_ollama import ChatOllama
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)

from langchain_core.messages import (HumanMessage, AIMessage)

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human" , "{question}")
])

chain = prompt | llm

history = [
    HumanMessage(content="Hi"),
    AIMessage(content="Hello!"),
    HumanMessage(content="My name is Akash"),
    AIMessage(content="Nice to meet you.")
]

response = chain.invoke({
    "history" : history,
    "question" : "What is my name?"
})

print(response.content)