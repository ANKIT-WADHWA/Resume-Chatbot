import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load API keys
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GROQ_API_KEY or not GOOGLE_API_KEY:
    print("❌ Error: Missing API keys! Please check your .env file.")
    exit()

# Initialize Llama-3
llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)

# Initialize Google AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

def load_and_split_pdf(pdf_path):
    """Loads a PDF file and splits it into smaller chunks."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Extract name from first page
    first_page_text = docs[0].page_content if docs else ""
    
    # Extract name heuristically
    name = "Ankit Wadhwa" if "Ankit Wadhwa" in first_page_text else "Unknown"

    # Splitting into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    return split_docs, name

def create_faiss_index(documents):
    """Creates a FAISS vector store from the documents."""
    texts = [doc.page_content for doc in documents]
    return FAISS.from_texts(texts, embeddings)

def retrieve_similar_results(vector_store, query):
    """Finds the most similar documents to the given query."""
    return vector_store.similarity_search(query, k=3)

def ask_llama(query, context):
    """Generates an answer using Llama-3 given the retrieved context."""
    if not context.strip():
        return "I couldn't find relevant information in the document."

    prompt = f"""
    You are an AI assistant. Answer the following question based on the provided document:

    📄 Document Content:
    {context}

    ❓ Question: {query}

    💡 Provide a concise and relevant response based only on the document.
    """

    response = llm.invoke(prompt)
    return response.content.strip()

# Main execution
if __name__ == "__main__":
    pdf_path = r"C:\Users\ankit\Downloads\Ankit_Wadhwa__Resume.pdf"

    # Load & extract name
    documents, extracted_name = load_and_split_pdf(pdf_path)

    # Create FAISS vector store
    vector_store = create_faiss_index(documents)

    while True:
        user_query = input("\n🔹 Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("👋 Exiting...")
            break

        # Handle name-specific queries directly
        if "name" in user_query.lower():
            print("\n🔹 Answer:\n")
            print(f"The name mentioned in the document is: {extracted_name}")
            continue

        # Retrieve similar documents
        similar_docs = retrieve_similar_results(vector_store, user_query)

        # Pass retrieved context to Llama-3
        context_text = "\n\n".join([doc.page_content for doc in similar_docs])
        answer = ask_llama(user_query, context_text)

        print("\n🔹 Answer:\n")
        print(answer)
