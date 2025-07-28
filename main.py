#!/usr/bin/env python3
"""
Adobe India Hackathon 2025 - Round 1B: Persona-Driven Document Intelligence
High-performance Python solution for intelligent PDF processing and summarization
"""

import os
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity


class DocumentIntelligence:
    def __init__(self, models_dir: str = "./models"):
        """Initialize the Document Intelligence system with local models."""
        self.models_dir = Path(models_dir)
        self.embedding_model = None
        self.summarizer = None
        self.tokenizer = None
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load all required models from local storage."""
        print("Loading models...")
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(exist_ok=True)
        
        try:
            # Load sentence transformer for embeddings
            embedding_path = self.models_dir / "all-MiniLM-L6-v2"
            if embedding_path.exists() and any(embedding_path.iterdir()):
                print("Loading existing embedding model...")
                self.embedding_model = SentenceTransformer(str(embedding_path))
            else:
                print("Downloading embedding model...")
                self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                self.embedding_model.save(str(embedding_path))
            
            # Load T5 model for summarization
            t5_path = self.models_dir / "t5-small"
            if t5_path.exists() and any(t5_path.iterdir()):
                print("Loading existing T5 model...")
                self.tokenizer = T5Tokenizer.from_pretrained(str(t5_path))
                model = T5ForConditionalGeneration.from_pretrained(str(t5_path))
                self.summarizer = pipeline("summarization", 
                                         model=model, 
                                         tokenizer=self.tokenizer,
                                         device=-1)  # CPU only
            else:
                print("Downloading T5 model...")
                self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
                model = T5ForConditionalGeneration.from_pretrained('t5-small')
                self.tokenizer.save_pretrained(str(t5_path))
                model.save_pretrained(str(t5_path))
                self.summarizer = pipeline("summarization", 
                                         model=model, 
                                         tokenizer=self.tokenizer,
                                         device=-1)
            
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to load required models: {str(e)}")
    
    def extract_pdf_content(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract meaningful content chunks from PDF using improved parsing."""
        content_chunks = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if not text.strip():
                    continue
                
                # Split text into meaningful paragraphs and sections
                paragraphs = self._extract_meaningful_paragraphs(text)
                
                for para in paragraphs:
                    if len(para['content'].split()) >= 10:  # Only meaningful content
                        content_chunks.append({
                            'document': os.path.basename(pdf_path),
                            'page_number': page_num + 1,
                            'section_title': para['title'],
                            'content': para['content'],
                            'word_count': len(para['content'].split())
                        })
            
            doc.close()
            
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
        
        return content_chunks
    
    def _extract_meaningful_paragraphs(self, text: str) -> List[Dict[str, str]]:
        """Extract meaningful paragraphs with proper titles."""
        paragraphs = []
        lines = text.split('\n')
        
        current_content = ""
        current_title = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line could be a title/header
            if self._is_potential_title(line):
                # Save previous paragraph if it has substantial content
                if current_content.strip() and len(current_content.split()) >= 10:
                    paragraphs.append({
                        'title': current_title or self._generate_title_from_content(current_content),
                        'content': current_content.strip()
                    })
                
                current_title = line
                current_content = ""
            else:
                current_content += line + " "
        
        # Add the last paragraph
        if current_content.strip() and len(current_content.split()) >= 10:
            paragraphs.append({
                'title': current_title or self._generate_title_from_content(current_content),
                'content': current_content.strip()
            })
        
        return paragraphs
    
    def _is_potential_title(self, line: str) -> bool:
        """Improved title detection."""
        line = line.strip()
        
        if len(line) < 3 or len(line) > 100:
            return False
        
        # Common title patterns
        title_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction"
            r'^[A-Z][A-Z\s]{2,}$',  # "INTRODUCTION"
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$',  # "Introduction"
            r'^\d+\.\d+',  # "1.1 Background"
            r'^(Abstract|Introduction|Conclusion|References|Methodology|Results|Discussion|Overview|Summary)$',
            r'^Chapter\s+\d+',
            r'^Section\s+\d+',
            r'^[A-Z][a-z]+(\s+(and|or|of|in|for|with|to)\s+[A-Z][a-z]+)*$',  # "Tips and Tricks"
        ]
        
        for pattern in title_patterns:
            if re.match(pattern, line):
                return True
        
        # Heuristic: short lines that start with capital and don't end with punctuation
        words = line.split()
        if (len(words) <= 8 and 
            line[0].isupper() and 
            not line.endswith(('.', ',', ';', ':', '!', '?')) and
            not any(word.islower() and len(word) > 3 for word in words[:2])):  # Avoid sentences
            return True
        
        return False
    
    def _generate_title_from_content(self, content: str) -> str:
        """Generate a title from content if no explicit title found."""
        words = content.split()[:8]
        title = ' '.join(words)
        if len(title) > 50:
            title = title[:47] + "..."
        return title
    
    def rank_content_by_persona_and_job(self, content_chunks: List[Dict], persona: str, job_to_be_done: str, top_k: int = 5) -> List[Dict]:
        """Rank content by relevance to persona, job, and PDF file names using semantic similarity."""
        if not content_chunks:
            return []
        
        # Create a combined query that includes both persona and job context
        persona_job_query = f"As a {persona}, I need to {job_to_be_done}"
        
        # Create embeddings for the persona-job query
        query_embedding = self.embedding_model.encode([persona_job_query])
        
        # Create embeddings for all content chunks with enhanced context
        content_texts = []
        for chunk in content_chunks:
            # Extract meaningful keywords from PDF filename
            pdf_name = chunk['document']
            filename_context = self._extract_filename_context(pdf_name)
            
            # Combine filename context, title, and content for better matching
            combined_text = f"{filename_context} {chunk['section_title']} {chunk['content']}"
            content_texts.append(combined_text)
        
        content_embeddings = self.embedding_model.encode(content_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, content_embeddings)[0]
        
        # Add similarity scores and filename relevance
        for i, chunk in enumerate(content_chunks):
            base_similarity = float(similarities[i])
            
            # Calculate filename relevance boost
            filename_boost = self._calculate_filename_relevance(
                chunk['document'], persona, job_to_be_done
            )
            
            # Combine similarity with filename relevance (weighted)
            chunk['similarity_score'] = base_similarity + (filename_boost * 0.3)  # 30% weight for filename
            chunk['filename_relevance'] = filename_boost
        
        # Sort by combined score (descending) and take top_k
        ranked_content = sorted(content_chunks, key=lambda x: x['similarity_score'], reverse=True)
        
        # Add importance rank
        for i, chunk in enumerate(ranked_content[:top_k]):
            chunk['importance_rank'] = i + 1
        
        return ranked_content[:top_k]
    
    def _extract_filename_context(self, pdf_filename: str) -> str:
        """Extract meaningful context from PDF filename."""
        # Remove file extension and clean filename
        name = pdf_filename.replace('.pdf', '').replace('.PDF', '')
        
        # Replace common separators with spaces
        name = name.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        
        # Split into words and filter out common non-meaningful words
        words = name.split()
        meaningful_words = []
        
        skip_words = {'ideas', 'tips', 'guide', 'manual', 'document', 'file', 'pdf', 'part', 'section'}
        
        for word in words:
            word_clean = word.lower().strip()
            if len(word_clean) > 2 and word_clean not in skip_words:
                meaningful_words.append(word_clean)
        
        return ' '.join(meaningful_words)
    
    def _calculate_filename_relevance(self, pdf_filename: str, persona: str, job_to_be_done: str) -> float:
        """Calculate how relevant the PDF filename is to the persona and job."""
        filename_lower = pdf_filename.lower()
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        relevance_score = 0.0
        
        # Extract keywords from filename
        filename_keywords = self._extract_filename_context(pdf_filename).split()
        
        # Check for direct matches with persona keywords
        persona_keywords = persona_lower.split()
        for p_word in persona_keywords:
            if len(p_word) > 3:  # Skip short words
                for f_word in filename_keywords:
                    if p_word in f_word or f_word in p_word:
                        relevance_score += 0.4
        
        # Check for direct matches with job keywords
        job_keywords = job_lower.split()
        for j_word in job_keywords:
            if len(j_word) > 3:  # Skip short words like 'a', 'the', 'of'
                for f_word in filename_keywords:
                    if j_word in f_word or f_word in j_word:
                        relevance_score += 0.5
        
        # Special keyword matching for common domains
        domain_keywords = {
            'food': ['breakfast', 'lunch', 'dinner', 'meal', 'recipe', 'cooking', 'vegetarian', 'vegan'],
            'travel': ['travel', 'trip', 'vacation', 'destination', 'hotel', 'flight', 'tourism'],
            'business': ['business', 'corporate', 'company', 'management', 'strategy', 'finance'],
            'health': ['health', 'medical', 'wellness', 'fitness', 'nutrition', 'diet'],
            'education': ['education', 'learning', 'study', 'academic', 'research', 'university']
        }
        
        # Check if filename contains domain-specific keywords that match persona/job
        for domain, keywords in domain_keywords.items():
            if domain in persona_lower or domain in job_lower:
                for keyword in keywords:
                    if keyword in filename_lower:
                        relevance_score += 0.3
        
        # Normalize score to 0-1 range
        return min(relevance_score, 1.0)
    
    def generate_persona_specific_summaries(self, content_chunks: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
        """Generate persona-specific summaries with enhanced quality metrics and intelligent formatting."""
        summaries = []
        
        for chunk in content_chunks:
            try:
                # Create persona-specific prompt for summarization
                content = chunk['content']
                pdf_filename = chunk.get('document', '')
                
                # Persona-specific processing with filename context
                refined_text = self._create_persona_specific_summary(content, persona, job_to_be_done, pdf_filename)
                
                summaries.append({
                    'document': chunk['document'],
                    'page_number': chunk['page_number'],
                    'refined_text': refined_text
                })
                
            except Exception as e:
                print(f"Error creating summary for {chunk.get('document', 'unknown')}: {str(e)}")
                # Enhanced fallback with intelligent formatting
                content = chunk.get('content', '')
                fallback_text = content[:300] + "..." if len(content) > 300 else content
                formatted_fallback = self._intelligent_text_formatter(fallback_text)
                
                summaries.append({
                    'document': chunk.get('document', 'unknown'),
                    'page_number': chunk.get('page_number', 1),
                    'refined_text': formatted_fallback
                })
        
        return summaries
    
    def _generate_dynamic_keywords(self, persona: str, job_to_be_done: str) -> List[str]:
        """Dynamically generate keywords based on persona and job description using NLP analysis."""
        keywords = set()
        
        # Extract keywords from persona
        persona_words = persona.lower().split()
        for word in persona_words:
            if len(word) > 2:  # Skip very short words
                keywords.add(word)
        
        # Extract keywords from job description
        job_words = job_to_be_done.lower().split()
        for word in job_words:
            if len(word) > 3:  # Skip short words for job description
                keywords.add(word)
        
        # Add semantic expansions based on common professional contexts
        semantic_expansions = {
            'planner': ['planning', 'schedule', 'organize', 'coordination', 'timeline', 'itinerary'],
            'manager': ['management', 'leadership', 'team', 'strategy', 'operations', 'oversight'],
            'doctor': ['medical', 'health', 'patient', 'treatment', 'diagnosis', 'clinical'],
            'researcher': ['research', 'analysis', 'study', 'data', 'methodology', 'findings'],
            'contractor': ['contract', 'service', 'delivery', 'project', 'requirements', 'specifications'],
            'scientist': ['scientific', 'analysis', 'experiment', 'data', 'methodology', 'results'],
            'professional': ['professional', 'expertise', 'standards', 'quality', 'compliance'],
            'consultant': ['consulting', 'advisory', 'recommendations', 'solutions', 'expertise'],
            'analyst': ['analysis', 'data', 'insights', 'trends', 'evaluation', 'metrics'],
            'specialist': ['specialized', 'expertise', 'technical', 'specific', 'focused'],
            'prepare': ['preparation', 'planning', 'setup', 'organize', 'arrange'],
            'create': ['creation', 'develop', 'design', 'build', 'generate'],
            'manage': ['management', 'control', 'oversee', 'coordinate', 'supervise'],
            'analyze': ['analysis', 'evaluation', 'assessment', 'review', 'examination'],
            'develop': ['development', 'creation', 'building', 'design', 'implementation'],
            'food': ['nutrition', 'dietary', 'ingredients', 'recipes', 'cooking', 'meal'],
            'travel': ['tourism', 'destination', 'accommodation', 'transportation', 'itinerary'],
            'business': ['corporate', 'commercial', 'enterprise', 'organization', 'company'],
            'health': ['wellness', 'medical', 'fitness', 'healthcare', 'treatment'],
            'education': ['learning', 'academic', 'study', 'training', 'knowledge'],
            'technology': ['technical', 'digital', 'software', 'systems', 'innovation'],
            'finance': ['financial', 'monetary', 'budget', 'cost', 'investment'],
            'legal': ['law', 'compliance', 'regulation', 'policy', 'procedure'],
            'marketing': ['promotion', 'advertising', 'branding', 'campaign', 'outreach']
        }
        
        # Add semantic expansions for matching words
        for base_word, expansions in semantic_expansions.items():
            if base_word in ' '.join([persona.lower(), job_to_be_done.lower()]):
                keywords.update(expansions)
        
        # Convert to list and limit to reasonable number
        keyword_list = list(keywords)[:15]  # Limit to 15 most relevant keywords
        
        return keyword_list
    
    def _intelligent_text_formatter(self, text: str) -> str:
        """Intelligently format text with proper capitalization, punctuation, and readability."""
        if not text or len(text.strip()) == 0:
            return text
        
        # Clean and normalize text
        text = text.strip()
        
        # Split into sentences and process each
        sentences = text.split('.')
        formatted_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 2:
                continue
            
            # Capitalize first letter
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
            # Fix common formatting issues
            sentence = sentence.replace(' ,', ',')
            sentence = sentence.replace(' .', '.')
            sentence = sentence.replace('  ', ' ')
            
            # Ensure proper spacing after punctuation
            sentence = sentence.replace(',', ', ').replace('  ', ' ')
            sentence = sentence.replace(':', ': ').replace('  ', ' ')
            
            formatted_sentences.append(sentence)
        
        # Join sentences with proper punctuation
        result = '. '.join(formatted_sentences)
        
        # Ensure proper ending
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
    def _calculate_content_confidence(self, content: str, persona: str, job_to_be_done: str) -> float:
        """Calculate confidence score for content relevance to persona and job."""
        if not content:
            return 0.0
        
        content_lower = content.lower()
        persona_lower = persona.lower()
        job_lower = job_to_be_done.lower()
        
        # Get dynamic keywords
        keywords = self._generate_dynamic_keywords(persona, job_to_be_done)
        
        # Calculate various relevance metrics
        keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
        persona_word_matches = sum(1 for word in persona_lower.split() if word in content_lower and len(word) > 2)
        job_word_matches = sum(1 for word in job_lower.split() if word in content_lower and len(word) > 3)
        
        # Normalize scores
        keyword_score = min(keyword_matches / max(len(keywords), 1), 1.0)
        persona_score = min(persona_word_matches / max(len(persona_lower.split()), 1), 1.0)
        job_score = min(job_word_matches / max(len(job_lower.split()), 1), 1.0)
        
        # Weighted confidence calculation
        confidence = (keyword_score * 0.4 + persona_score * 0.3 + job_score * 0.3)
        
        return round(confidence, 3)
    
    def _enhance_content_quality(self, content: str, persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """Enhance content with quality metrics and intelligent processing."""
        if not content:
            return {
                "enhanced_text": "",
                "confidence_score": 0.0,
                "quality_metrics": {
                    "readability": "poor",
                    "relevance": "low",
                    "completeness": "incomplete"
                }
            }
        
        # Format text intelligently
        enhanced_text = self._intelligent_text_formatter(content)
        
        # Calculate confidence
        confidence = self._calculate_content_confidence(content, persona, job_to_be_done)
        
        # Quality metrics
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Determine quality ratings
        readability = "excellent" if word_count > 20 and sentence_count > 1 else "good" if word_count > 10 else "fair"
        relevance = "high" if confidence > 0.6 else "medium" if confidence > 0.3 else "low"
        completeness = "complete" if word_count > 15 and sentence_count > 1 else "partial" if word_count > 5 else "minimal"
        
        return {
            "enhanced_text": enhanced_text,
            "confidence_score": confidence,
            "quality_metrics": {
                "readability": readability,
                "relevance": relevance,
                "completeness": completeness,
                "word_count": word_count,
                "sentence_count": sentence_count
            }
        }
    
    def _create_persona_specific_summary(self, content: str, persona: str, job_to_be_done: str, pdf_filename: str = "") -> str:
        """Create a summary tailored to the specific persona, job requirements, and PDF context using T5 model."""
        
        # Generate focus keywords dynamically based on persona and job
        focus_keywords = self._generate_dynamic_keywords(persona, job_to_be_done)
        
        # Extract filename context for additional relevance
        filename_context = self._extract_filename_context(pdf_filename) if pdf_filename else ""
        filename_keywords = filename_context.split()
        
        # Clean and prepare content for summarization
        content = content.strip()
        if len(content) < 50:
            return content  # Too short to summarize meaningfully
        
        # Try T5 summarization with persona and filename context first
        try:
            # Limit content length for T5 model
            max_content_length = 400
            if len(content) > max_content_length:
                # Extract most relevant parts first
                sentences = content.split('.')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 10:
                        continue
                    
                    # Score sentence based on persona relevance, job keywords, and filename context
                    score = 0
                    sentence_lower = sentence.lower()
                    job_lower = job_to_be_done.lower()
                    
                    # Check for persona-specific keywords
                    for keyword in focus_keywords:
                        if keyword in sentence_lower:
                            score += 2
                    
                    # Check for job-specific keywords
                    job_words = job_lower.split()
                    for word in job_words:
                        if len(word) > 3 and word in sentence_lower:
                            score += 3
                    
                    # Check for filename-related keywords (new)
                    for f_keyword in filename_keywords:
                        if len(f_keyword) > 3 and f_keyword in sentence_lower:
                            score += 2
                    
                    relevant_sentences.append((sentence, score))
                
                # Sort by relevance and take top sentences that fit within limit
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                
                selected_content = ""
                for sentence, score in relevant_sentences:
                    if len(selected_content + sentence) < max_content_length:
                        selected_content += sentence + ". "
                    else:
                        break
                
                content = selected_content.strip() if selected_content else content[:max_content_length]
            
            # Create summarization prompt with persona and filename context
            context_info = f"{persona}"
            if filename_context:
                context_info += f" working with {filename_context}"
            
            input_text = f"summarize for {context_info}: {content}"
            
            # Generate summary using T5
            summary_result = self.summarizer(
                input_text, 
                max_length=min(150, len(content.split()) + 20),  # Dynamic max length
                min_length=20, 
                do_sample=False,
                truncation=True
            )
            
            summary = summary_result[0]['summary_text']
            
            # Post-process summary to ensure it's persona-relevant and filename-aware
            if summary and len(summary) > 10:
                # Apply intelligent formatting for proper capitalization and readability
                summary = self._intelligent_text_formatter(summary)
                return summary
            else:
                raise Exception("Summary too short")
                
        except Exception as e:
            # Fallback: Create a manual summary from most relevant sentences
            sentences = content.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 15:
                    continue
                
                # Score sentence based on persona relevance, job keywords, and filename context
                score = 0
                sentence_lower = sentence.lower()
                job_lower = job_to_be_done.lower()
                
                # Check for persona-specific keywords
                for keyword in focus_keywords:
                    if keyword in sentence_lower:
                        score += 2
                
                # Check for job-specific keywords
                job_words = job_lower.split()
                for word in job_words:
                    if len(word) > 3 and word in sentence_lower:
                        score += 3
                
                # Check for filename-related keywords (new)
                for f_keyword in filename_keywords:
                    if len(f_keyword) > 3 and f_keyword in sentence_lower:
                        score += 2
                
                if score > 0:
                    relevant_sentences.append((sentence, score))
            
            # Sort by relevance score and create summary
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            if relevant_sentences:
                # Take top 2-3 most relevant sentences and create a coherent summary
                top_sentences = [sent[0] for sent in relevant_sentences[:2]]
                summary = '. '.join(top_sentences)
                # Apply intelligent formatting for proper capitalization and readability
                summary = self._intelligent_text_formatter(summary)
                return summary
            else:
                # Final fallback with intelligent formatting
                fallback_content = content[:150] + "..." if len(content) > 150 else content
                return self._intelligent_text_formatter(fallback_content)
    
    def process_documents(self, pdf_files: List[Path], persona: str, job_to_be_done: str) -> Dict[str, Any]:
        """Main processing pipeline with persona-specific analysis."""
        start_time = time.time()
        
        if not pdf_files:
            print(f"Warning: No PDF files provided")
            # Return empty result structure
            return {
                "metadata": {
                    "input_documents": [],
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }
        
        print(f"Processing {len(pdf_files)} PDF files")
        
        # Extract content from all PDFs
        all_content = []
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"Processing ({i}/{len(pdf_files)}): {pdf_file.name}")
            content_chunks = self.extract_pdf_content(str(pdf_file))
            all_content.extend(content_chunks)
        
        print(f"Extracted {len(all_content)} content chunks")
        
        # Rank content by persona and job relevance
        top_content = self.rank_content_by_persona_and_job(all_content, persona, job_to_be_done, top_k=5)
        
        print(f"Found {len(top_content)} relevant sections")
        
        # Generate persona-specific summaries
        summaries = self.generate_persona_specific_summaries(top_content, persona, job_to_be_done)
        
        # Prepare output in the exact challenge format
        result = {
            "metadata": {
                "input_documents": [pdf.name for pdf in pdf_files],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": chunk['document'],
                    "section_title": chunk['section_title'],
                    "importance_rank": chunk['importance_rank'],
                    "page_number": chunk['page_number']
                }
                for chunk in top_content
            ],
            "subsection_analysis": [
                {
                    "document": summary['document'],
                    "refined_text": summary['refined_text'],
                    "page_number": summary['page_number']
                }
                for summary in summaries
            ]
        }
        
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return result


def main():
    """Main execution function."""
    print("Adobe India Hackathon 2025 - Round 1B")
    print("Persona-Driven Document Intelligence")
    print("=" * 50)
    
    # Configuration
    import sys
    
    # Allow specifying input JSON file as command line argument
    if len(sys.argv) > 1:
        INPUT_JSON_FILE = sys.argv[1]
    else:
        INPUT_JSON_FILE = "./app/input/challenge1.json"
    
    PDF_DIR = "./app/PDFs"
    OUTPUT_DIR = "./app/output"
    MODELS_DIR = "./models"
    
    # Create directories if they don't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Read input from JSON file
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"Successfully loaded input from: {INPUT_JSON_FILE}")
    except FileNotFoundError:
        print(f"ERROR: Input JSON file not found: {INPUT_JSON_FILE}")
        print("Please ensure the input JSON file exists.")
        print("Usage: python main.py [path_to_input_json]")
        return 1
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format in {INPUT_JSON_FILE}: {str(e)}")
        return 1
    
    # Extract persona and job from JSON
    try:
        persona = input_data["persona"]["role"]
        job_to_be_done = input_data["job_to_be_done"]["task"]
        pdf_documents = input_data["documents"]
        
        print(f"Persona: {persona}")
        print(f"Job to be done: {job_to_be_done}")
        print(f"Documents to process: {len(pdf_documents)}")
        print()
        
    except KeyError as e:
        print(f"ERROR: Missing required field in JSON: {str(e)}")
        return 1
    
    # Check if PDF directory exists
    pdf_path = Path(PDF_DIR)
    if not pdf_path.exists():
        print(f"ERROR: PDF directory not found: {PDF_DIR}")
        print("Please ensure the PDFs folder exists.")
        return 1
    
    # Get the specific PDF files mentioned in the JSON
    pdf_files = []
    missing_files = []
    
    for doc in pdf_documents:
        pdf_filename = doc["filename"]
        pdf_file_path = pdf_path / pdf_filename
        
        if pdf_file_path.exists():
            pdf_files.append(pdf_file_path)
            print(f"   ✓ Found: {pdf_filename}")
        else:
            missing_files.append(pdf_filename)
            print(f"   ✗ Missing: {pdf_filename}")
    
    if missing_files:
        print(f"\nWARNING: {len(missing_files)} PDF files not found in {PDF_DIR}")
        for missing in missing_files:
            print(f"   - {missing}")
        
        if not pdf_files:
            print("ERROR: No PDF files found. Cannot proceed.")
            return 1
        else:
            print(f"\nProceeding with {len(pdf_files)} available PDF files...")
    
    print()
    print("Starting document processing...")
    print("=" * 50)
    
    try:
        # Initialize the system
        print("Initializing Document Intelligence...")
        doc_intel = DocumentIntelligence(models_dir=MODELS_DIR)
        
        # Process documents
        result = doc_intel.process_documents(pdf_files, persona, job_to_be_done)
        
        # Create a unique output filename based on persona and job
        persona_clean = "".join(c for c in persona if c.isalnum() or c in (' ', '-', '_')).rstrip()
        persona_clean = persona_clean.replace(' ', '_').lower()
        
        # Get first few words of job for filename
        job_words = job_to_be_done.split()[:3]  # First 3 words
        job_clean = "_".join(word.lower() for word in job_words if word.isalnum())
        
        # Create timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output filename
        output_filename = f"{persona_clean}_{job_clean}_{timestamp}.json"
        output_file = Path(OUTPUT_DIR) / output_filename
        
        # Save result
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print()
        print("Processing completed successfully!")
        print("=" * 50)
        print(f"Results saved to: {output_file}")
        print(f"Documents processed: {len(result['metadata']['input_documents'])}")
        print(f"Relevant sections found: {len(result['extracted_sections'])}")
        print(f"Summaries generated: {len(result['subsection_analysis'])}")
        print()
        print("Document intelligence analysis complete!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())