import os
import tempfile
import pickle
import shutil
import logging

from fastapi import UploadFile

# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

DB_PATH = "vectorstores/db_faiss"
SPLIT_DOCS_PATH = os.path.join(DB_PATH, "split_docs.pkl")
EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PolicyManager:
    """
    Manages the processing of policy documents and the creation/update
    of the vector store knowledge base.
    """
    def __init__(self, db_path=DB_PATH):
        """
        Initializes the PolicyManager with the path to the vector store.
        """
        self.db_path = db_path
        self.split_docs_path = os.path.join(self.db_path, "split_docs.pkl")
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=ollama_base_url)
        
        os.makedirs(self.db_path, exist_ok=True)

    def update_vector_store(self, file: UploadFile):
        """
        The main public method to update the knowledge base from a PDF file.
        This function orchestrates the entire process.
        """
        logger.info(f"Starting policy update with file: {file.filename}")
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                shutil.copyfileobj(file.file, tmp_file)
                temp_file_path = tmp_file.name

            split_docs = self._load_and_split_pdf(temp_file_path)
            
            if not split_docs:
                logger.warning(f"No text could be extracted from {file.filename}.")
                return {"success": False, "error": "No text could be extracted from the provided PDF."}

            self._create_and_save_stores(split_docs)

            logger.info("✅ Vector store and policy documents updated successfully.")
            return {"success": True, "message": "Policy updated successfully"}

        except Exception as e:
            logger.error(f"❌ Failed to update policy: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.info(f"Cleaned up temporary file: {temp_file_path}")


    def _load_and_split_pdf(self, pdf_path: str):
        """
        Loads a PDF document and splits it into manageable chunks for processing.
        """
        logger.info(f"Loading and splitting PDF from {pdf_path}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
        except Exception as e:
            logger.warning(f"PyPDFLoader failed: {e}. Falling back to PyMuPDFLoader.")
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"Document split into {len(split_docs)} chunks.")
        return split_docs

    def _create_and_save_stores(self, split_docs: list):
        """
        Creates embeddings, builds the FAISS vector store, and saves both the
        store and the raw split documents for the retrievers.
        """
        logger.info("Creating embeddings and building vector store...")
        
        db = FAISS.from_documents(split_docs, self.embeddings)
        
        db.save_local(self.db_path)
        logger.info(f"FAISS vector store saved to {self.db_path}")

        with open(self.split_docs_path, 'wb') as f:
            pickle.dump(split_docs, f)
        logger.info(f"Raw split documents saved to {self.split_docs_path}")

def build_initial_store(pdf_path: str):
    """
    A command-line helper function to build the vector store for the first time.
    This is extremely useful for development and testing.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found at {pdf_path}")
        return

    print("="*50)
    print("🛠️  Building Initial Knowledge Base...")
    print(f"Source PDF: {pdf_path}")
    print("="*50)

    class MockUploadFile:
        def __init__(self, path):
            self.filename = os.path.basename(path)
            self.file = open(path, 'rb')
        
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.file.close()

    try:
        with MockUploadFile(pdf_path) as mock_file:
            manager = PolicyManager()
            result = manager.update_vector_store(mock_file)
    except Exception as e:
        result = {"success": False, "error": str(e)}


    if result["success"]:
        print("\n✅ Knowledge base built successfully!")
    else:
        print(f"\n❌ Error building knowledge base: {result['error']}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python policy_manager.py <path_to_policy_pdf>")
        print("Example: python policy_manager.py 'Policies/Glucose Monitor - Policy Article.pdf'")
    else:
        build_initial_store(sys.argv[1])