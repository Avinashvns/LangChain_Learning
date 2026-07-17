from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
vector = embeddings.embed_query("What is Python?")
print(vector)
print(len(vector))

# Multiple text

vectors= embeddings.embed_documents(["Python","Java"])
print(len(vectors[1]))