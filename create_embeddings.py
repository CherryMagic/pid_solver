"""
Creates embeddings from multiple PDFs using sentence-level chunks for better precision
"""

import os
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
import time

class EmbeddingCreator:
    def __init__(self, pdf_paths=None, output_dir="rag_data"):
        """
        Initialize with multiple PDF paths
        
        Args:
            pdf_paths: List of PDF paths to process
            output_dir: Directory to save embeddings and chunks
        """
        if pdf_paths is None:
            self.pdf_paths = [
                "data/eduka_4.pdf",
                "data/kreativni_centar_4.pdf", 
                "data/zavod_4.pdf"
            ]
        else:
            self.pdf_paths = pdf_paths
            
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Store all chunks from all PDFs
        self.all_chunks = []
        self.all_sentences = []
        self.source_info = {}  # Track which PDF each chunk comes from
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a single PDF"""
        print(f"\n Reading PDF: {pdf_path}")
        
        if not os.path.exists(pdf_path):
            print(f"PDF not found: {pdf_path}")
            return "", []
        
        reader = PdfReader(pdf_path)
        text = ""
        pages = []
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty pages
                pages.append({
                    'page_num': i+1,
                    'text': page_text,
                    'source': os.path.basename(pdf_path)
                })
                text += page_text + "\n"
        
        print(f"   Loaded {len(pages)} pages from {os.path.basename(pdf_path)}")
        return text, pages
    
    def extract_all_texts(self):
        """Extract text from all PDFs"""
        print("\n" + "="*60)
        print("READING ALL PDFS")
        print("="*60)
        
        all_text = ""
        all_pages = []
        total_pages = 0
        
        for pdf_path in self.pdf_paths:
            text, pages = self.extract_text_from_pdf(pdf_path)
            if text:
                all_text += f"\n--- SOURCE: {os.path.basename(pdf_path)} ---\n{text}"
                all_pages.extend(pages)
                total_pages += len(pages)
        
        print(f"\n Total loaded: {total_pages} pages from {len(self.pdf_paths)} PDFs")
        return all_text, all_pages
    
    def split_into_sentences(self, text):
        """Split text into sentences (better than arbitrary word chunks)"""
        # First split by common sentence endings
        # Pattern for Serbian sentence endings (., !, ?) followed by space and capital letter
        sentence_endings = r'(?<=[.!?])\s+(?=[А-ШЂЈЋЖЉЊA-Z])'
        sentences = re.split(sentence_endings, text)
        
        # Clean up and filter very short sentences
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
        return sentences
    
    def create_chunks(self, all_text, all_pages):
        """Create chunks at sentence level from all PDFs"""
        print(f"\n Splitting into sentences and chunks...")
        
        # First split into sentences from the combined text
        all_sentences = self.split_into_sentences(all_text)
        print(f"Total sentences: {len(all_sentences)}")
        
        # Group sentences into small chunks (2-3 sentences per chunk)
        chunk_size = 3  # sentences per chunk
        chunks = []
        sentence_chunks = []
        
        # Create a mapping to find which page/source each sentence belongs to
        # This is approximate but works for most cases
        for i in range(0, len(all_sentences), chunk_size):
            chunk_sentences = all_sentences[i:i+chunk_size]
            chunk_text = ' '.join(chunk_sentences)
            
            # Try to find which source this chunk belongs to
            source = "unknown"
            page_num = 1
            
            # Simple heuristic: look for source markers in the chunk
            for page in all_pages:
                if page['text'][:100] in chunk_text or chunk_text[:100] in page['text']:
                    source = page['source']
                    page_num = page['page_num']
                    break
            
            chunks.append({
                'id': len(chunks),
                'source': source,
                'page': page_num,
                'text': chunk_text,
                'sentences': len(chunk_sentences),
                'length': len(chunk_text)
            })
        
        # Also create single-sentence chunks for exact matching
        for i, sentence in enumerate(all_sentences):
            # Find source for this sentence
            source = "unknown"
            page_num = 1
            for page in all_pages:
                if sentence[:100] in page['text'] or page['text'][:100] in sentence:
                    source = page['source']
                    page_num = page['page_num']
                    break
            
            sentence_chunks.append({
                'id': i,
                'source': source,
                'page': page_num,
                'text': sentence,
                'sentences': 1,
                'length': len(sentence)
            })
        
        print(f"\n Chunk statistics:")
        print(f"   - Chunks (2-3 sentences): {len(chunks)}")
        print(f"   - Individual sentences: {len(sentence_chunks)}")
        
        return chunks, sentence_chunks
    
    def create_embeddings(self, chunks, sentence_chunks):
        """Create embeddings using djovak/embedic-large"""
        print(f"\n Creating embeddings...")
        
        start = time.time()
        
        # Load the Serbian-specialized model
        model_name = "djovak/embedic-large"
        print(f" Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Create embeddings for chunks (2-3 sentences)
        chunk_texts = [chunk['text'] for chunk in chunks]
        print(f"\n Creating embeddings for {len(chunk_texts)} chunks...")
        
        batch_size = 16
        chunk_embeddings = []
        
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i+batch_size]
            batch_emb = self.model.encode(batch, show_progress_bar=True)
            chunk_embeddings.append(batch_emb)
            print(f"Chunks: {min(i+batch_size, len(chunk_texts))}/{len(chunk_texts)}")
        
        chunk_embeddings = np.vstack(chunk_embeddings)
        
        # Create embeddings for single sentences
        sentence_texts = [sent['text'] for sent in sentence_chunks]
        print(f"\n Creating embeddings for {len(sentence_texts)} individual sentences...")
        
        sentence_embeddings = []
        for i in range(0, len(sentence_texts), batch_size):
            batch = sentence_texts[i:i+batch_size]
            batch_emb = self.model.encode(batch, show_progress_bar=True)
            sentence_embeddings.append(batch_emb)
        
        sentence_embeddings = np.vstack(sentence_embeddings)
        
        elapsed = time.time() - start
        print(f"\n Time: {elapsed:.1f} seconds")
        
        return chunk_embeddings, sentence_embeddings
    
    def save_visible_chunks(self, chunks, sentence_chunks):
        """Save chunks in a human-readable format"""
        
        # Save readable chunks (2-3 sentences)
        with open(f"{self.output_dir}/chunks_review.txt", 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHUNKS REVIEW (2-3 SENTENCES)\n")
            f.write("="*80 + "\n\n")
            
            # Group chunks by source
            by_source = {}
            for chunk in chunks:
                source = chunk['source']
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(chunk)
            
            for source, source_chunks in by_source.items():
                f.write(f"\n{'#'*60}\n")
                f.write(f"SOURCE: {source}\n")
                f.write(f"{'#'*60}\n\n")
                
                for chunk in source_chunks:
                    f.write(f"\n{'-'*60}\n")
                    f.write(f"CHUNK #{chunk['id']} (Page {chunk['page']}, {chunk['sentences']} sentences, {chunk['length']} chars)\n")
                    f.write(f"{'-'*60}\n")
                    f.write(chunk['text'])
                    f.write("\n\n")
        
        # Save readable sentences
        with open(f"{self.output_dir}/sentences_review.txt", 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("INDIVIDUAL SENTENCES REVIEW\n")
            f.write("="*80 + "\n\n")
            
            # Group sentences by source
            by_source = {}
            for sent in sentence_chunks:
                source = sent['source']
                if source not in by_source:
                    by_source[source] = []
                by_source[source].append(sent)
            
            for source, source_sents in by_source.items():
                f.write(f"\n{'#'*60}\n")
                f.write(f"SOURCE: {source}\n")
                f.write(f"{'#'*60}\n\n")
                
                for sent in source_sents:
                    f.write(f"\n{'-'*40}\n")
                    f.write(f"SENTENCE #{sent['id']} (Page {sent['page']})\n")
                    f.write(f"{'-'*40}\n")
                    f.write(sent['text'])
                    f.write("\n\n")
        
        print(f"\n Review files saved:")
        print(f" - {self.output_dir}/chunks_review.txt")
        print(f" - {self.output_dir}/sentences_review.txt")
    
    def run(self):
        """Run the complete pipeline"""
        print("="*60)
        print("CREATING EMBEDDINGS FOR ALL PDFS")
        print("="*60)
        print(f"\nPDFs being processed:")
        for pdf in self.pdf_paths:
            print(f"- {pdf}")
        
        # Extract text from all PDFs
        all_text, all_pages = self.extract_all_texts()
        
        if not all_text:
            print("\n No texts to process. Check if PDFs exist.")
            return
        
        # Create chunks
        chunks, sentence_chunks = self.create_chunks(all_text, all_pages)
        
        # Create embeddings
        chunk_embeddings, sentence_embeddings = self.create_embeddings(chunks, sentence_chunks)
        
        # Save everything
        print(f"\n Saving data...")
        
        # Save chunks
        with open(f"{self.output_dir}/chunks.pkl", 'wb') as f:
            pickle.dump(chunks, f)
        
        # Save embeddings
        np.save(f"{self.output_dir}/embeddings.npy", chunk_embeddings)
        
        # Save sentences
        with open(f"{self.output_dir}/sentences.pkl", 'wb') as f:
            pickle.dump(sentence_chunks, f)
        
        np.save(f"{self.output_dir}/sentence_embeddings.npy", sentence_embeddings)
        
        # Save metadata
        metadata = {
            'num_chunks': len(chunks),
            'num_sentences': len(sentence_chunks),
            'embedding_dim': chunk_embeddings.shape[1],
            'model': 'djovak/embedic-large',
            'sources': list(set([c['source'] for c in chunks]))
        }
        with open(f"{self.output_dir}/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n - Data saved in directory: {self.output_dir}")
        print(f" - chunks.pkl: {len(chunks)} chunks")
        print(f" - sentences.pkl: {len(sentence_chunks)} sentences")
        print(f" - embeddings.npy: {chunk_embeddings.shape}")
        
        # Save visible chunks for inspection
        self.save_visible_chunks(chunks, sentence_chunks)
        
        print("\n" + "="*60)
        print("COMPLETE! All chunks are ready.")
        print("="*60)
        print(f"\nStatistics:")
        print(f" - Total chunks (2-3 sentences): {len(chunks)}")
        print(f" - Total individual sentences: {len(sentence_chunks)}")
        print(f" - Embedding dimension: {chunk_embeddings.shape[1]}")
        print(f"\n Review files created:")
        print(f" - {self.output_dir}/chunks_review.txt")
        print(f" - {self.output_dir}/sentences_review.txt")
        print(f"\n Run 'rag_answers.py' to start asking questions")

if __name__ == "__main__":
    # You can specify custom PDF paths here if needed
    pdfs = [
        "data/eduka_4.pdf",
        "data/kreativni_centar_4.pdf", 
        "data/zavod_4.pdf"
    ]
    
    creator = EmbeddingCreator(pdf_paths=pdfs)
    creator.run()