from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2")

prompt = PromptTemplate.from_template(
    "What is {topic}?"
)

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({"topic" : "Python"})

print(response)