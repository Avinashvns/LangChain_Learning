from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(
    model = "llama3.2"
)

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple language."
)

chain = prompt | llm

response = chain.invoke({"topic" : "Python"})

print(response.content)