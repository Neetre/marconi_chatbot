import os
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import logging
from .qdrant_manager import QdrantManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchoolChatbot:
    def __init__(self):
        # Initialize vector database manager
        self.vector_db = QdrantManager(
            collection_name="school_knowledge_base",
            embedding_model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize LLM
        model_name = "TheBloke/Phi-2-GPTQ"  # Example, choose appropriate model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                load_in_4bit=True,  # Quantization for efficiency
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
            )
            logger.info("LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
        
        # Conversation history storage
        self.conversations = {}
        
    def answer_question(self, question: str, conversation_id: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """Process a question and generate an answer with reference sources"""
        # Get conversation history if available
        conversation_history = self._get_conversation_history(conversation_id)
        
        # Retrieve relevant documents using the QdrantManager
        relevant_docs = self.vector_db.retrieve_documents(question, n_results=5)
        
        # Construct prompt with context and conversation history
        prompt = self._construct_prompt(question, relevant_docs, conversation_history)
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Update conversation history
        if conversation_id:
            self._update_conversation(conversation_id, question, response)
        
        # Extract sources from relevant docs
        sources = [{"title": doc["metadata"].get("title", ""), "url": doc["metadata"].get("url", "")} 
                  for doc in relevant_docs]
        
        return response, sources
    
    def _construct_prompt(self, question: str, documents: List[Dict], 
                          conversation_history: List[Dict] = None) -> str:
        """Construct a prompt with context for the LLM"""
        # Start with system instructions
        prompt = """You are a helpful school assistant chatbot. Answer the question based on the provided context.
If you don't know the answer, say so - don't make up information.
Provide clear, concise responses that are helpful to students, parents, or staff.

Context information:
"""
        
        # Add context from retrieved documents
        for doc in documents:
            prompt += f"---\n{doc['text']}\n"
        
        prompt += "---\n\n"
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt += "Previous conversation:\n"
            for exchange in conversation_history[-3:]:  # Include last 3 exchanges
                prompt += f"User: {exchange['user']}\n"
                prompt += f"Assistant: {exchange['assistant']}\n"
        
        # Add the current question
        prompt += f"\nCurrent question: {question}\n\nAnswer: "
        
        return prompt
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response using the LLM"""
        try:
            response = self.generator(prompt)[0]['generated_text']
            # Extract just the answer part (after "Answer: ")
            answer = response.split("Answer: ")[-1].strip()
            return answer
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I'm sorry, but I encountered an error processing your question. Please try again."
    
    def _get_conversation_history(self, conversation_id: Optional[str]) -> List[Dict]:
        """Get conversation history for a given ID"""
        if not conversation_id:
            return []
        return self.conversations.get(conversation_id, [])
    
    def _update_conversation(self, conversation_id: str, question: str, answer: str):
        """Update the conversation history"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "user": question,
            "assistant": answer
        })
        
        # Limit history to last 10 exchanges to prevent context growth
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]