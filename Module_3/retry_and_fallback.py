from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Fallback

primary_llm = ChatOllama(model="llama3.2").with_retry(stop_after_attempt=3)
backup_llm = ChatOllama(model = "qwen:4b")
llm = primary_llm.with_fallbacks([backup_llm])

# retry
# llm = ChatOllama(model="llama3.2").with_retry(stop_after_attempt=3)

prompt = ChatPromptTemplate.from_template(
"""
Answer the following question:

{question}
"""
)

chain = (
    prompt | llm | StrOutputParser()
)

answer = chain.invoke({
    "question" : "What is Python?"
})

print(answer)
