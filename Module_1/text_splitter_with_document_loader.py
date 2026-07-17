from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

loader = PyPDFLoader("Module_1/python.pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)
print(chunks)

# chunks ko Pandas DataFrame

data= []
for i, chunk in enumerate(chunks):
    data.append({
        "Chunk No": i + 1,
        "Page": chunk.metadata["page"],
        "Source": chunk.metadata["source"],
        "Content": chunk.page_content
    })
df = pd.DataFrame(data)
print(df)
