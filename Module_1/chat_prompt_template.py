from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system" , "You are a helpful AI teacher."),
    ("human" , "Explain {topic} in simple language.")
])

formatted_prompt = prompt.invoke({"topic":"Python"})
print(formatted_prompt)