from pathlib import Path
import json
from typing import List, Dict
import argparse
from .qdrant_manager import QdrantManager

def load_documents(data_dir: str) -> List[Dict]:
    """Load documents from a directory of JSON files"""
    documents = []
    data_path = Path(data_dir)
    
    for file_path in data_path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Expect format: {"text": "...", "metadata": {"title": "...", ...}}
                if isinstance(data, list):
                    documents.extend(data)
                else:
                    documents.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    return documents

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into chunks of approximately chunk_size characters with overlap"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        # Find a good breaking point (end of sentence) near chunk_size
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # Try to find the end of a sentence
            while end > start + chunk_size - 200 and end < len(text) and text[end] not in ['.', '!', '?', '\n']:
                end += 1
            if end < len(text) and text[end] in ['.', '!', '?', '\n']:
                end += 1
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def index_documents(documents: List[Dict], collection_name: str = "school_knowledge_base"):
    """Index documents into Qdrant"""
    # Initialize Qdrant manager
    qdrant_manager = QdrantManager(collection_name=collection_name)
    
    # Process and index documents
    chunk_counter = 0
    texts = []
    metadatas = []
    ids = []
    
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        
        # Chunk the document
        chunks = chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            # Create a unique ID for each chunk
            chunk_id = f"{metadata.get('id', 'doc')}_{chunk_counter}"
            chunk_counter += 1
            
            # Create metadata for the chunk
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            
            # Add to collection
            texts.append(chunk)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
            
            # Process in batches of 100 to avoid memory issues
            if len(texts) >= 100:
                qdrant_manager.add_documents(texts, metadatas, ids)
                texts = []
                metadatas = []
                ids = []
    
    # Add any remaining documents
    if texts:
        qdrant_manager.add_documents(texts, metadatas, ids)
            
    print(f"Indexed {chunk_counter} chunks from {len(documents)} documents")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index school documents for the chatbot")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing JSON documents")
    args = parser.parse_args()
    
    documents = load_documents(args.data_dir)
    index_documents(documents)
    print("Indexing complete!")