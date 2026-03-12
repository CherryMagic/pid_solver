"""
Use small chunks for precise retrieval
"""

import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import time

class RAGAnswerSystem:
    def __init__(self, data_dir="rag_data"):
        print("="*60)
        print(" LOADING RAG SYSTEM (SMALL CHUNKS)")
        print("="*60)
        
        self.data_dir = data_dir
        self.load_data()
        self.load_model()
        
    def load_data(self):
        """Load pre-computed chunks and embeddings"""
        print("\n Loading the data...")
        
        # Load chunks (2-3 sentences)
        with open(f"{self.data_dir}/chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        print(f" Loaded {len(self.chunks)} chunks")
        
        # Load embeddings
        self.embeddings = np.load(f"{self.data_dir}/embeddings.npy")
        print(f" Loaded embeddings with dimension {self.embeddings.shape}")
        
        # Load single sentences for exact matching
        with open(f"{self.data_dir}/sentences.pkl", 'rb') as f:
            self.sentences = pickle.load(f)
        print(f" Loaded {len(self.sentences)} single sentences")
        
        # Load sentence embeddings
        self.sentence_embeddings = np.load(f"{self.data_dir}/sentence_embeddings.npy")
        print(f" Loaded embeddings of sentences with dimension {self.sentence_embeddings.shape}")
        
        # Load metadata
        with open(f"{self.data_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load embedding model
        print("\n Loading Embedić modela...")
        self.embed_model = SentenceTransformer(self.metadata['model'])
        print(f" Model: {self.metadata['model']}")
    
    def load_model(self):
        """Load Llama for answering"""
        print("\n Loading Llama modela...")
        
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.llm = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            print(" Llama model loaded")
            
        except Exception as e:
            print(f" Error: {e}")
            self.llm = None
    
    def find_relevant_chunks(self, question, top_k=5):
        """Find most relevant chunks - first try sentences, then chunks"""
        
        # Encode question
        q_embedding = self.embed_model.encode([question])
        
        # First search in single sentences (most precise)
        sentence_similarities = np.dot(self.sentence_embeddings, q_embedding.T).flatten()
        sentence_indices = np.argsort(sentence_similarities)[-top_k:][::-1]
        
        sentence_results = []
        for idx in sentence_indices:
            if sentence_similarities[idx] > 0.3:  # Only keep relevant ones
                sentence_results.append({
                    'type': 'sentence',
                    'chunk': self.sentences[idx],
                    'score': float(sentence_similarities[idx]),
                    'text': self.sentences[idx]['text']
                })
        
        # If we have good sentence matches, return them
        if len(sentence_results) >= 3:
            print(f"\n Found {len(sentence_results)} precise sentences")
            return sentence_results[:top_k]
        
        # Otherwise search in larger chunks
        chunk_similarities = np.dot(self.embeddings, q_embedding.T).flatten()
        chunk_indices = np.argsort(chunk_similarities)[-top_k:][::-1]
        
        chunk_results = []
        for idx in chunk_indices:
            chunk_results.append({
                'type': 'chunk',
                'chunk': self.chunks[idx],
                'score': float(chunk_similarities[idx]),
                'text': self.chunks[idx]['text']
            })
        
        print(f"\nFound {len(chunk_results)} relevant chunks")
        return chunk_results
    
    def generate_answer(self, question, relevant_chunks):
        """Generate answer using a very simple, direct prompt"""
        if not self.llm:
            return "The model is not loaded."
        
        # Just take the most relevant chunk's text
        if not relevant_chunks:
            return "I have no information about that."
        
        best = relevant_chunks[0]
        context = best['text'][:500]  # Limit context length
        
        # Ultra-simple prompt
        prompt = f"""Tekst: {context}

            Pitanje: {question}

            Odgovor:"""
        
        response = self.llm(
            prompt,
            max_new_tokens=50,
            temperature=0.1,
            do_sample=False,
            return_full_text=False
        )[0]['generated_text']
        
        # Clean up
        answer = response.strip()
        if '\n' in answer:
            answer = answer.split('\n')[0]
        
        return answer
    
    def ask(self, question):
        """Ask a question and get answer with sources"""
        print(f"\n{'='*60}")
        print(f"Pitanje: {question}")
        print(f"{'='*60}")
        
        start = time.time()
        
        # Find relevant chunks
        relevant = self.find_relevant_chunks(question)
        
        if not relevant:
            print("\nNo relevant information in the document.")
            return
        
        print(f"\nFound {len(relevant)} relevant parts:")
        for i, r in enumerate(relevant):
            source_type = "SENTENCE" if r['type'] == 'sentence' else "CHUNK"
            print(f"\n{source_type} #{r['chunk']['id']} (Page {r['chunk']['page']}) ---")
            print(f"Relevatnost: {r['score']:.3f}")
            
            # Highlight if it contains the exact word from question
            question_words = question.lower().split()
            text_lower = r['text'].lower()
            for word in question_words:
                if len(word) > 3 and word in text_lower:
                    print(f"Contains the word: '{word}'")
            
            print(f"Text: {r['text'][:200]}...")
        
        # Generate answer
        print(f"\nGenerating answer...")
        answer = self.generate_answer(question, relevant)
        
        elapsed = time.time() - start
        
        print(f"\nANSWER:")
        print(f"{answer}")
        print(f"\nTime: {elapsed:.1f} seconds")
        
        return {
            'question': question,
            'answer': answer,
            'sources': relevant,
            'time': elapsed
        }
    
    def interactive(self):
        """Interactive Q&A session"""
        print("\n" + "="*60)
        print("🎓 SERBIAN RAG SYSTEM (SMALL CHUNKS)")
        print("="*60)
        print("\nEnter a question in Serbian (or 'kraj' for exit)")
        print("-" * 60)
        
        while True:
            question = input("\n Your question: ").strip()
            
            if question.lower() in ['kraj', 'izlaz', 'exit', 'quit']:
                print("👋 Goodbye!")
                break
            
            if question:
                self.ask(question)

def main():
    rag = RAGAnswerSystem()
    rag.interactive()

if __name__ == "__main__":
    main()