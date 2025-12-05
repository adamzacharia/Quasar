import os
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class RAGService:
    """Service for Retrieval Augmented Generation"""
    
    def __init__(self, persist_directory: str = None):
        if persist_directory is None:
            # Default to project_root/chroma_db
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(root_dir, "chroma_db")
            
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        
        # Initialize vector store if it exists
        if os.path.exists(persist_directory):
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )

    def ingest_document(self, file_path: str) -> bool:
        """Ingest a PDF document into the vector store"""
        try:
            print(f"Loading {file_path}...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            print(f"Splitting {len(documents)} pages...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            print(f"Creating embeddings for {len(chunks)} chunks...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            # self.vector_store.persist() # Chroma 0.4+ persists automatically
            print("Ingestion complete.")
            return True
        except Exception as e:
            print(f"Ingestion failed: {e}")
            return False

    def search(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant documents"""
        if not self.vector_store:
            return []
            
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Search failed: {e}")
            return []
