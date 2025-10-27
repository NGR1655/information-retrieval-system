import os
import time
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# ✅ Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ✅ Extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    print("📄 Starting PDF text extraction...")
    start_time = time.time()

    for pdf in pdf_docs:
        print(f"Reading: {pdf.name}")
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            print("❌ Error reading PDF:", e)

    print(f"✅ Extraction finished in {round(time.time() - start_time, 2)} seconds")
    print(f"🧾 Extracted text length: {len(text)} characters")
    return text


# ✅ Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        print("⚠️ Warning: No chunks created. PDF might be empty or unreadable.")
    return chunks


# ✅ Create FAISS Vector Store with batching
def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("⚠️ No text chunks found. Check if PDF was read correctly.")

    print("🧠 Creating vector store using Google embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    all_vectors = []
    batch_size = 100  # Google API limit

    try:
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            print(f"🔹 Embedding batch {i // batch_size + 1} ({len(batch)} chunks)...")
            batch_vector = FAISS.from_texts(batch, embedding=embeddings)
            all_vectors.append(batch_vector)

        
        if len(all_vectors) > 1:
            print("🔗 Merging FAISS batches...")
            base_store = all_vectors[0]
            for other_store in all_vectors[1:]:
                base_store.merge_from(other_store)
            vector_store = base_store
        else:
            vector_store = all_vectors[0]

        print("✅ Vector store created successfully.")
        return vector_store

    except Exception as e:
        print("❌ Embedding error:", e)
        print("⚙️ Please check your API quota, key, or billing permissions.")
        raise



def get_conversational_chain(vector_store):
    try:
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        print("✅ Using model: gemini-2.5-flash")
    except Exception as e:
        print("⚠️ gemini-2.5-flash not available, falling back to gemini-2.5-pro")
        print("Error:", e)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )
    print("✅ Conversational chain ready.")
    return conversation_chain














