from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load llm Model
llm = ChatOllama(model="llama3.2")

# prompt
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple language."
)

# Output Parser
parser = StrOutputParser()

# Create Chain
chain = prompt | llm | parser

while True:

    topic = input("\nEnter Topic : ")

    if topic.lower() == "exit":
        break

    response = chain.invoke(
        {
            "topic" : topic
        }
    )

    print("\nAnswer : \n")
    print(response)