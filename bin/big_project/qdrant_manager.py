import os
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, collection_name: str = "school_knowledge_base", embedding_model_name: str = "all-MiniLM-L6-v2"):
        """Initialize Qdrant manager with a collection name and embedding model
        
        Args:
            collection_name: Name of the Qdrant collection
            embedding_model_name: Name of the sentence transformer model to use for embeddings
        """
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        logger.info(f"Initialized embedding model: {embedding_model_name}")
        
        # Initialize Qdrant client
        # Try to connect to local Qdrant instance first, fallback to in-memory
        try:
            self.client = QdrantClient(url="localhost", port=6333)
            logger.info("Connected to local Qdrant instance")
        except Exception as e:
            logger.warning(f"Could not connect to local Qdrant: {e}. Using in-memory instance.")
            self.client = QdrantClient(":memory:")
            
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
        
    def _create_collection_if_not_exists(self):
        """Create the collection if it doesn't already exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                # Create a new collection with the same dimensionality as the embedding model
                vector_size = self.embedding_model.get_sentence_embedding_dimension()
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error while creating collection: {e}")
            raise
            
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector database
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dictionaries for each document
            ids: Optional list of IDs for the documents
        """
        if not texts:
            logger.warning("No texts provided to add_documents")
            return
            
        # Generate embeddings for all texts
        embeddings = self.embedding_model.encode(texts)
        
        # Prepare points to add
        points = []
        for i in range(len(texts)):
            point_id = ids[i] if ids and i < len(ids) else str(i)
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # Add the text to the payload for retrieval
            payload = {"text": texts[i], **metadata}
            
            points.append(models.PointStruct(
                id=point_id,
                vector=embeddings[i].tolist(),
                payload=payload
            ))
            
        # Add to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        logger.info(f"Added {len(points)} documents to collection {self.collection_name}")
        
    def retrieve_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve relevant documents for a query
        
        Args:
            query: The query text
            n_results: Number of results to return
        
        Returns:
            List of dictionaries containing retrieved documents with text and metadata
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query)
        
        # Search for similar vectors
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=n_results
        )
        
        # Format results
        results = []
        for result in search_results:
            payload = result.payload
            text = payload.pop("text", "")
            results.append({
                "id": result.id,
                "text": text,
                "metadata": payload,
                "score": result.score
            })
            
        return results