import os
import glob
import warnings
from flask import Flask, render_template, request, jsonify

# Ollama & LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# Silence those pesky Pydantic warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# --- GLOBAL AI SETUP ---
# This part runs once when the server starts
print("--- Initializing AI and Loading PDF ---")

# 1. Find PDF
pdf_files = glob.glob("data/*.pdf")
if not pdf_files:
    raise FileNotFoundError("No PDF found in the 'data' folder!")

selected_pdf = pdf_files[0]
print(f"--- Processing: {selected_pdf} ---")

# 2. Load PDF & Create Vector Store
loader = PyPDFLoader(selected_pdf)
pages = loader.load_and_split()

# Using llama3.2 for local embeddings and processing
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = FAISS.from_documents(pages, embeddings)

# 3. Setup the Local LLM (Brain)
llm = OllamaLLM(model="llama3.2")

# --- WEB ROUTES ---

@app.route('/')
def index():
    # This renders your templates/index.html
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query')
    
    if not user_query:
        return jsonify({"answer": "Please enter a question."})

    try:
        # Search for relevant text in the PDF
        docs = vectorstore.similarity_search(user_query, k=3)
        context = "\n".join([d.page_content for d in docs])

        # Formulate the prompt for Llama 3.2
        prompt = f"""
        You are a professional AI assistant. 
        Answer the question strictly using the provided context.
        
        CONTEXT:
        {context}
        
        QUESTION: 
        {user_query}
        
        ANSWER:
        """
        
        # Get response from local Ollama
        response = llm.invoke(prompt)
        return jsonify({"answer": response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"answer": "Sorry, I encountered an error processing that."})

if __name__ == '__main__':
    # Server will run on http://127.0.0.1:5000
    app.run(debug=True)