import pandas as pd

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# CSV Tool

@tool
def read_csv(question: str) -> str:
    """
    Read employee.csv and answer basic questions.
    """

    df = pd.read_csv("Module_4/employee.csv")

    question = question.lower()

    if "total" in question or "count" in question:

        return str(len(df))

    elif "highest salary" in question:

        employee = df.loc[df["salary"].idxmax()]

        return employee.to_json()

    elif "average salary" in question:

        return str(df["salary"].mean())

    elif "departments" in question:

        return ", ".join(df["department"].unique())

    else:

        return df.to_string(index=False)
    
# LLM
llm = ChatOllama(
    model = "llama3.2",
    temperature=0
)

llm = llm.bind_tools([read_csv])

# Chat Loop
while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    response = llm.invoke(question)

    if response.tool_calls:

        tool_call = response.tool_calls[0]

        result = read_csv.invoke(tool_call["args"])

        print("\nCSV Result :\n")

        print(result)

    else:

        print(response.content)