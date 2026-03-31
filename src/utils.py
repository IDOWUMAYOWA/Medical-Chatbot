from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


## Load the PDF files from the directory
def load_pdfs_from_directory(directory_path):
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    return documents
def filter_to_key_contents(docs: List[Document]) -> List[Document]:
    """
    Reduce metadata in a list of LangChain Document objects.

    This function creates a new list of Document objects that retain the
    original `page_content` but keep only the `"source"` field from the
    metadata (if it exists). All other metadata fields are removed.

    If a document does not contain `"source"` in its metadata (or it is None/empty),
    the returned document will have an empty metadata dictionary.

    Args:
        docs (List[Document]): A list of LangChain Document objects.

    Returns:
        List[Document]: A new list of Document objects containing the same
        page_content with minimal metadata.
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        source = doc.metadata.get("source")
        metadata = {"source": source} if source else {}

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata
            )
        )

    return minimal_docs    
# Split the documents into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=25,
    )
    texts_chunk = text_splitter.split_documents(extracted_data)
    return texts_chunk

#Download the Embeddings from HuggingFace 
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')  
    return embeddings