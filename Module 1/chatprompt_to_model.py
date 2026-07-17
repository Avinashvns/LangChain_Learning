from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOllama(model="llama3.2")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI teacher."),
    ("human", "Explain {topic} in simple language.")
])

chain = prompt | llm

response = chain.invoke({"topic": "Python"})

print(response.content)