from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple language."
)

final_prompt = prompt.invoke({"topic" : "Python"})
print(final_prompt.text)