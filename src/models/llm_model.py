"""
LLM Model Module

Handles response generation using Ollama's local models.
"""

import os
import ollama
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class LLMModel:
    def __init__(self, model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
    
    def generate_response(self, query: str, context: str, detected_language: str = "en") -> str:
        """Generate a response using the LLM."""
        try:
            # Construct the prompt with context and language instruction
            if detected_language == "ar":
                # Make the instruction the absolute first part of the prompt
                prompt = "Your response MUST be in Arabic only.\n\n" \
                         "You are a helpful AI assistant. Use the following context to answer the question.\n" \
                         "If you cannot find the answer in the context, say so.\n\n" \
                         f"Context: {context}\n\n" \
                         f"Question: {query}\n\n" \
                         "Answer:"
            else:
                prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say so.

Context: {context}

Question: {query}

Answer:"""
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()["response"]
        
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def generate_responses(self, queries: List[str], contexts: List[str]) -> List[str]:
        """Generate responses for multiple queries."""
        return [
            self.generate_response(query, context)
            for query, context in zip(queries, contexts)
        ]

    def generate_response_with_context(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate response using the LLM with context from vector search.
        
        Args:
            query: User's question
            context: List of relevant documents from vector search
            
        Returns:
            Generated response
        """
        # Prepare context
        context_text = "\n\n".join([
            f"Document {i+1}:\n{result['text']}"
            for i, result in enumerate(context)
        ])
        
        # Prepare prompt
        prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.
If you cannot find the answer in the context, say so.

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            # Generate response
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            return response["response"]
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response." 