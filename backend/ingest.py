import os
import warnings
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
# NEW IMPORT:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Silence the LangChain deprecation warnings so the console stays clean
warnings.filterwarnings("ignore")

def build_knowledge_layer():
    docs = []
    
    # 1. Parse all ASIC PDFs in the directory
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf") and f.startswith("asic")]
    for pdf_path in pdf_files:
        print(f"Loading PDF: {pdf_path}...")
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())

    # 2. Parse all ASIC Excel files
    excel_files = [f for f in os.listdir(".") if f.endswith(".xlsx") and f.startswith("asic")]
    for excel_path in excel_files:
        print(f"Loading Excel: {excel_path}...")
        df = pd.read_excel(excel_path)
        for index, row in df.iterrows():
            content = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            docs.append(Document(page_content=content, metadata={"source": excel_path, "row": index}))

    if not docs:
        print("No documents found to index.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, 
        chunk_overlap=150, 
        separators=["\n\nArticle", "\n\nSection", "\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Generating embeddings for {len(chunks)} chunks...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
    print("Ingestion complete! Local ChromaDB populated.")

if __name__ == "__main__":
    build_knowledge_layer()