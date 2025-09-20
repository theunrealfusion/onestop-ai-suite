import os
from dotenv import load_dotenv
load_dotenv()

# Ensure knowledge_base folder exists and has at least one document
knowledge_base_dir = "./knowledge_base"
os.makedirs(knowledge_base_dir, exist_ok=True)
sample_doc_path = os.path.join(knowledge_base_dir, "france_rag.txt")
if not os.path.exists(sample_doc_path):
    with open(sample_doc_path, "w", encoding="utf-8") as f:
        f.write(
            "France is a country in Western Europe. Its capital city is Paris, which is known for its art, culture, and history. "
            "Paris is home to landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.\n\n"
            "RAG (Retrieval-Augmented Generation) is an approach that combines retrieval of relevant documents with generative models "
            "to answer questions using both context and model knowledge."
        )

# Load documents
documents = []
for filename in os.listdir(knowledge_base_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(knowledge_base_dir, filename), "r", encoding="utf-8") as f:
            documents.append(f.read())

# Select embedding provider: "openai", "gemini", or "anthropic"
embedding_provider = "gemini"  # Change as needed

import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()
collection = client.get_or_create_collection(name="rag_demo")

embeddings = []
if embedding_provider == "openai":
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    collection = client.create_collection(name="rag_demo", embedding_function=openai_ef)
    for i, doc in enumerate(documents):
        collection.add(documents=[doc], ids=[str(i)])

elif embedding_provider == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    for i, doc in enumerate(documents):
        doc_embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT")
        emb = doc_embedder.embed_query(doc)
        embeddings.append(emb)
        collection.add(documents=[doc], ids=[str(i)], embeddings=[emb])
        print(f"Added document {i} to collection: {doc[:40]}...")

elif embedding_provider == "anthropic":
    import anthropic
    client_a = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    for doc in documents:
        emb_response = client_a.embeddings.create(
            model="claude-3-haiku-20240307",
            input=doc
        )
        emb = emb_response.embedding
        embeddings.append(emb)
    for i, doc in enumerate(documents):
        collection.add(documents=[doc], ids=[str(i)], embeddings=[embeddings[i]])
else:
    print("Invalid embedding provider selected.")

# Step 4: Retrieve relevant chunks for a RAG-specific query
query = "How does RAG use retrieved context to answer questions about France?"
if embedding_provider == "gemini":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    query_embedder = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="RETRIEVAL_QUERY")
    query_embedding = query_embedder.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=2)
else:
    results = collection.query(query_texts=[query], n_results=2)
retrieved_chunks = results["documents"][0]
print("Retrieved Chunks:", retrieved_chunks)

# Step 5: Pass context + query to LLM for answer generation
llm_provider = "gemini"  # Change to "openai" or "anthropic" as needed

context = "\n".join(retrieved_chunks)
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
print("Prompt:", query)

if llm_provider == "openai":
    import openai
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    print("OpenAI Response:", response.choices[0].message["content"])

elif llm_provider == "anthropic":
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    print("Claude Response:", response.content[0].text)

elif llm_provider == "gemini":
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    response = model.generate_content(prompt)
    print("Gemini Response:", response.text)
else:
    print("Invalid LLM provider selected.")