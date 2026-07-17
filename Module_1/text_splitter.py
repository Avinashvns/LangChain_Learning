from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


text= """
    Python is a programming language.
    It is easy to learn.
    It is used in AI and ML.
"""

splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=30,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print(chunks)

recur_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100
)

chunks_recursive = recur_splitter.split_text(text)
print("Recursive : ",chunks_recursive)