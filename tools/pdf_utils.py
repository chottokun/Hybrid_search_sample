import pdfplumber
import os
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

def extract_texts_from_pdfs(
    pdf_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    all_chunks: List[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False, # Default, can be adjusted if needed
    )
    for fname in os.listdir(pdf_dir):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, fname)
            try:
                with pdfplumber.open(path) as pdf:
                    full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    if full_text.strip():
                        # Split the extracted text into chunks
                        texts = text_splitter.split_text(full_text)
                        for i, text_chunk in enumerate(texts):
                            # Create a Document for each chunk with metadata
                            chunk_doc = Document(
                                page_content=text_chunk,
                                metadata={"source": fname, "chunk_index": i}
                            )
                            all_chunks.append(chunk_doc)
            except Exception as e:
                print(f"[WARN] {fname} 読み込み失敗: {e}")
    return all_chunks
