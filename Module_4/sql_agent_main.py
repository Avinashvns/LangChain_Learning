from langchain_core.tools import tool
import sqlite3
from langchain_ollama import ChatOllama

@tool
def execute_sql(query: str) -> str:
    """
    Execute SQL query.
    """

    conn = sqlite3.connect("Module_4/company.db")

    cursor = conn.cursor()

    cursor.execute(query)

    result = cursor.fetchall()

    conn.close()

    return str(result)


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

llm = llm.bind_tools(
    [execute_sql]
)

# Chat

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    response = llm.invoke(question)

    if response.tool_calls:

        tool_call = response.tool_calls[0]

        print("\nGenerated SQL :")

        print(tool_call["args"]["query"])

        result = execute_sql.invoke(
            tool_call["args"]
        )

        print("\nDatabase Result :")

        print(result)

    else:

        print(response.content)