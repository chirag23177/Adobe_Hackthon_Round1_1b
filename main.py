#!/usr/bin/env python3
"""
Document Intelligence System - Main Pipeline
===========================================

A CPU-only, offline document intelligence system that extracts and ranks 
relevant sections from PDF documents based on persona and job-to-be-done.

Author: Adobe Hackathon Team
Date: July 28, 2025
"""

import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Core dependencies
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Set PyTorch to CPU-only mode
torch.set_num_threads(4)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Configuration
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"
PDF_DIR = os.path.join(INPUT_DIR, "PDFs")
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
CHUNK_SIZE = 800  # Optimal chunk size for semantic coherence
TOP_K_SECTIONS = 5
MIN_SECTION_LENGTH = 50  # Minimum characters for a valid section


class DocumentProcessor:
    """Handles PDF parsing and text extraction with metadata."""
    
    def __init__(self):
        self.documents = {}
        
    def extract_text_from_pdf(self, pdf_path: str, doc_title: str) -> Dict[str, Any]:
        """
        Extract text content page-wise from a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            doc_title: Document title for metadata
            
        Returns:
            Dictionary with document metadata and page-wise content
        """
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # Clean and normalize text
                cleaned_text = self._clean_text(text)
                
                if len(cleaned_text) >= MIN_SECTION_LENGTH:
                    # Detect sections within the page
                    sections = self._detect_sections(cleaned_text, page_num + 1)
                    pages_content.extend(sections)
            
            doc.close()
            
            return {
                'filename': os.path.basename(pdf_path),
                'title': doc_title,
                'total_pages': len(doc),
                'sections': pages_content
            }
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return {'filename': os.path.basename(pdf_path), 'title': doc_title, 'sections': []}
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = re.sub(r'[^\w\s\-.,;:!?()"]', ' ', text)
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _detect_sections(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Detect logical sections within a page of text.
        
        Args:
            text: Page text content
            page_num: Page number
            
        Returns:
            List of section dictionaries with metadata
        """
        sections = []
        
        # Split by potential section headers (heuristic approach)
        # Look for patterns like: "Title:", "Chapter X", numbered sections, etc.
        section_patterns = [
            r'\n[A-Z][^.]*:',  # Lines ending with colon (likely headers)
            r'\n\d+\.\s+[A-Z]',  # Numbered sections
            r'\n[A-Z][A-Z\s]{10,50}\n',  # All caps headers
        ]
        
        split_points = [0]
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text))
            split_points.extend([m.start() for m in matches])
        
        split_points.append(len(text))
        split_points = sorted(set(split_points))
        
        # Create sections from split points
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            section_text = text[start:end].strip()
            
            if len(section_text) >= MIN_SECTION_LENGTH:
                # Extract potential title from first line
                lines = section_text.split('\n')
                title = self._extract_section_title(lines[0] if lines else "")
                
                # Create chunks if section is too long
                chunks = self._create_chunks(section_text, CHUNK_SIZE)
                
                for chunk_idx, chunk in enumerate(chunks):
                    section_title = f"{title}" if chunk_idx == 0 else f"{title} (Part {chunk_idx + 1})"
                    sections.append({
                        'title': section_title,
                        'content': chunk,
                        'page_number': page_num,
                        'section_index': len(sections),
                        'chunk_index': chunk_idx
                    })
        
        # If no sections detected, treat entire page as one section
        if not sections and len(text) >= MIN_SECTION_LENGTH:
            chunks = self._create_chunks(text, CHUNK_SIZE)
            for chunk_idx, chunk in enumerate(chunks):
                title = self._extract_section_title(text.split('\n')[0] if text else f"Page {page_num}")
                section_title = f"{title}" if chunk_idx == 0 else f"{title} (Part {chunk_idx + 1})"
                sections.append({
                    'title': section_title,
                    'content': chunk,
                    'page_number': page_num,
                    'section_index': chunk_idx,
                    'chunk_index': chunk_idx
                })
        
        return sections
    
    def _extract_section_title(self, first_line: str) -> str:
        """Extract a meaningful title from the first line of a section."""
        # Clean the line
        title = first_line.strip()
        # Remove common artifacts
        title = re.sub(r'^[\d\-\.\s]+', '', title)  # Remove leading numbers/dashes
        title = re.sub(r':$', '', title)  # Remove trailing colon
        # Limit length
        if len(title) > 80:
            title = title[:77] + "..."
        return title if title else "Untitled Section"
    
    def _create_chunks(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters,
        preserving sentence boundaries when possible.
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.!?]+\s+', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_all_documents(self, document_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process all PDF documents and return structured content."""
        all_documents = {}
        
        for doc_info in document_list:
            filename = doc_info['filename']
            title = doc_info['title']
            pdf_path = os.path.join(PDF_DIR, filename)
            
            if os.path.exists(pdf_path):
                print(f"Processing: {filename}")
                doc_data = self.extract_text_from_pdf(pdf_path, title)
                all_documents[filename] = doc_data
            else:
                print(f"Warning: File not found: {pdf_path}")
        
        return all_documents


class SemanticAnalyzer:
    """Handles embedding generation and similarity computation."""
    
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        # Force CPU usage
        self.model = SentenceTransformer(model_name, device='cpu')
        # Optimize for CPU inference
        self.model.eval()
        torch.set_grad_enabled(False)
        
    def create_query_embedding(self, persona: str, job_to_be_done: str) -> np.ndarray:
        """Create embedding for the search query (persona + job)."""
        query_text = f"Persona: {persona}. Task: {job_to_be_done}"
        return self.model.encode([query_text], convert_to_tensor=False)[0]
    
    def embed_sections(self, all_sections: List[Dict[str, Any]]) -> np.ndarray:
        """Generate embeddings for all document sections."""
        if not all_sections:
            return np.array([])
        
        # Prepare texts for embedding
        texts = []
        for section in all_sections:
            # Combine title and content for better semantic understanding
            combined_text = f"{section['title']}. {section['content']}"
            texts.append(combined_text)
        
        print(f"Generating embeddings for {len(texts)} sections...")
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def compute_similarities(self, query_embedding: np.ndarray, section_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and all sections."""
        if section_embeddings.size == 0:
            return np.array([])
        
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]
        return similarities


class SectionExtractor:
    """Handles section ranking and extraction of top relevant sections."""
    
    def __init__(self, top_k: int = TOP_K_SECTIONS):
        self.top_k = top_k
    
    def extract_top_sections(self, 
                           all_sections: List[Dict[str, Any]], 
                           similarities: np.ndarray,
                           documents: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract and rank the top K most relevant sections.
        
        Returns:
            List of top sections with metadata and ranking
        """
        if len(similarities) == 0:
            return []
        
        # Get indices of top K sections
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        extracted_sections = []
        
        for rank, idx in enumerate(top_indices, 1):
            section = all_sections[idx]
            similarity_score = similarities[idx]
            
            # Find the document this section belongs to
            doc_filename = None
            for filename, doc_data in documents.items():
                if any(s['section_index'] == section['section_index'] and 
                      s['page_number'] == section['page_number'] 
                      for s in doc_data['sections']):
                    doc_filename = filename
                    break
            
            extracted_section = {
                'document': doc_filename or "Unknown",
                'section_title': section['title'],
                'importance_rank': rank,
                'page_number': section['page_number'],
                'content': section['content'],
                'similarity_score': float(similarity_score)
            }
            
            extracted_sections.append(extracted_section)
        
        return extracted_sections


class SummaryGenerator:
    """Handles text refinement and summary generation."""
    
    def refine_section_content(self, content: str, max_length: int = 800) -> str:
        """
        Refine and clean section content for final output.
        
        Args:
            content: Raw section content
            max_length: Maximum length for refined content
            
        Returns:
            Cleaned and refined text
        """
        # Basic text cleaning and refinement
        refined = content.strip()
        
        # Ensure proper sentence structure
        sentences = re.split(r'[.!?]+\s*', refined)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                # Ensure sentence ends with proper punctuation
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                clean_sentences.append(sentence)
        
        refined_text = ' '.join(clean_sentences)
        
        # Truncate if too long while preserving sentence boundaries
        if len(refined_text) > max_length:
            sentences = refined_text.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated) + len(sentence) + 2 <= max_length:
                    truncated += sentence + '. '
                else:
                    break
            refined_text = truncated.strip()
        
        return refined_text
    
    def generate_subsection_analysis(self, top_sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate refined subsection analysis for the final output."""
        subsection_analysis = []
        
        for section in top_sections:
            refined_content = self.refine_section_content(section['content'])
            
            analysis = {
                'document': section['document'],
                'refined_text': refined_content,
                'page_number': section['page_number']
            }
            
            subsection_analysis.append(analysis)
        
        return subsection_analysis


def load_input_configuration() -> Tuple[Dict[str, Any], str, str, List[Dict[str, str]]]:
    """Load and parse the input JSON configuration."""
    input_path = os.path.join(INPUT_DIR, "challenge1b_input.json")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        persona = config['persona']['role']
        job_to_be_done = config['job_to_be_done']['task']
        documents = config['documents']
        
        return config, persona, job_to_be_done, documents
    
    except Exception as e:
        raise Exception(f"Error loading input configuration: {str(e)}")


def save_output_json(result: Dict[str, Any]) -> None:
    """Save the final result to output JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "challenge1b_output.json")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"Output saved to: {output_path}")
    except Exception as e:
        raise Exception(f"Error saving output: {str(e)}")


def main():
    """Main pipeline execution."""
    start_time = time.time()
    
    print("=" * 60)
    print("Document Intelligence System - Starting Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load input configuration
        print("\n1. Loading input configuration...")
        config, persona, job_to_be_done, document_list = load_input_configuration()
        print(f"   Persona: {persona}")
        print(f"   Task: {job_to_be_done}")
        print(f"   Documents: {len(document_list)} files")
        
        # Step 2: Process PDF documents
        print("\n2. Processing PDF documents...")
        processor = DocumentProcessor()
        all_documents = processor.process_all_documents(document_list)
        
        # Collect all sections from all documents
        all_sections = []
        for doc_data in all_documents.values():
            all_sections.extend(doc_data['sections'])
        
        print(f"   Extracted {len(all_sections)} sections from {len(all_documents)} documents")
        
        # Step 3: Generate embeddings
        print("\n3. Generating semantic embeddings...")
        analyzer = SemanticAnalyzer()
        
        # Create query embedding
        query_embedding = analyzer.create_query_embedding(persona, job_to_be_done)
        
        # Generate section embeddings
        section_embeddings = analyzer.embed_sections(all_sections)
        
        # Step 4: Compute similarities and rank sections
        print("\n4. Computing similarities and ranking sections...")
        similarities = analyzer.compute_similarities(query_embedding, section_embeddings)
        
        extractor = SectionExtractor()
        top_sections = extractor.extract_top_sections(all_sections, similarities, all_documents)
        
        print(f"   Selected top {len(top_sections)} most relevant sections")
        
        # Step 5: Generate refined summaries
        print("\n5. Generating refined summaries...")
        summarizer = SummaryGenerator()
        subsection_analysis = summarizer.generate_subsection_analysis(top_sections)
        
        # Step 6: Build final JSON output
        print("\n6. Building final JSON output...")
        
        # Prepare extracted_sections for output (without content and similarity_score)
        extracted_sections = []
        for section in top_sections:
            extracted_sections.append({
                'document': section['document'],
                'section_title': section['section_title'],
                'importance_rank': section['importance_rank'],
                'page_number': section['page_number']
            })
        
        # Build final result
        result = {
            'metadata': {
                'input_documents': [doc['filename'] for doc in document_list],
                'persona': persona,
                'job_to_be_done': job_to_be_done,
                'processing_timestamp': datetime.now().isoformat()
            },
            'extracted_sections': extracted_sections,
            'subsection_analysis': subsection_analysis
        }
        
        # Step 7: Save output
        print("\n7. Saving output...")
        save_output_json(result)
        
        # Performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print(f"Total processing time: {processing_time:.2f} seconds")
        print(f"Processed {len(all_documents)} documents with {len(all_sections)} sections")
        print(f"Selected {len(top_sections)} most relevant sections")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
