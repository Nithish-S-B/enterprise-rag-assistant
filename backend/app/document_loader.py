import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path: str) -> List[Document]:
    """
    Loads and parses a PDF document into a list of LangChain Document objects.
    
    Args:
        file_path (str): The path to the PDF file on disk.
        
    Returns:
        List[Document]: A list of LangChain Document objects, one per page.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file does not have a .pdf extension.
    """
    # Validate file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at '{file_path}' does not exist.")
        
    # Validate PDF extension (case-insensitive)
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("Invalid file format. Only PDF files (.pdf) are supported.")
        
    # Load and return the documents
    loader = PyPDFLoader(file_path)
    return loader.load()
