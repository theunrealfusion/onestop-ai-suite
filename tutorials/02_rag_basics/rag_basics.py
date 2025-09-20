import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
import openai

# Step 1: Load environment variables
load_dotenv()

# Step 2: Load knowledge base documents
knowledge_base_dir = "./tutorials/02_rag_basics/knowledge_base"
documents = []
for filename in os.listdir(knowledge_base_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(knowledge_base_dir, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Step 3: Create embeddings and store in ChromaDB
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)
client = chromadb.Client()
collection = client.create_collection(name="rag_demo", embedding_function=openai_ef)
for i, doc in enumerate(documents):
    collection.add(documents=[doc], ids=[str(i)])

# Step 4: Retrieve relevant chunks for a RAG-specific query
query = "How does RAG use retrieved context to answer questions about France?"
results = collection.query(query_texts=[query], n_results=2)
retrieved_chunks = results["documents"][0]
print("Retrieved Chunks:", retrieved_chunks)

# Step 5: Pass context + query to LLM for answer generation
context = "\n".join(retrieved_chunks)
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
print("LLM Response:", response.choices[0].message["content"])
