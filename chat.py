import os
import glob
import warnings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Hide Pydantic V1 warnings for a clean look
warnings.filterwarnings("ignore", category=UserWarning)

# 1. Setup
pdf_files = glob.glob("data/*.pdf")

if not pdf_files:
    print("No PDF found in /data!")
    exit()

print(f"--- Using PDF: {pdf_files[0]} ---")

# 2. Load and Index using Ollama (Llama 3.2)
# This uses your local Llama model to create the mathematical search index
print("--- Indexing PDF (this may take a minute locally) ---")
loader = PyPDFLoader(pdf_files[0])
pages = loader.load_and_split()

# Using llama3.2 for both searching and thinking
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(pages, embeddings)

# 3. Setup the Local LLM
llm = OllamaLLM(model="llama3.2")

print("--- Ready! Type 'exit' to quit ---")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]: break

    # Find the relevant pages locally
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n".join([d.page_content for d in docs])

    # Ask Llama 3.2
    prompt = f"Answer the question based ONLY on the context below:\n\n{context}\n\nQuestion: {query}"
    
    print("AI is thinking...")
    response = llm.invoke(prompt)
    print(f"\nAI: {response}\n")