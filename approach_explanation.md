# 🔍 Technical Approach - Document Intelligence System

## 🔍 Overview

The Document Intelligence System addresses the challenge of extracting and ranking relevant sections from a collection of PDF documents based on a specific persona and job-to-be-done. Given travel documents about South of France and a "Travel Planner" persona tasked with planning a 4-day trip for college friends, the system must intelligently identify the most relevant content sections and provide refined summaries in a structured JSON format.

This solution leverages semantic understanding through embeddings to move beyond simple keyword matching, enabling contextual relevance scoring that understands the relationship between content and user intent.

## 🏗️ Architecture

The system implements a three-phase pipeline designed for efficiency and accuracy:

### Phase 1: PDF Parsing & Preprocessing
- **Text Extraction**: Uses PyMuPDF for robust, page-wise text extraction from all PDF documents
- **Intelligent Sectioning**: Employs regex patterns to detect natural section breaks (headers, numbered lists, topic changes)
- **Content Chunking**: Splits large sections into 800-character chunks while preserving sentence boundaries for optimal semantic coherence
- **Metadata Collection**: Maintains document names, page numbers, and section titles throughout the pipeline

### Phase 2: Embedding-Based Semantic Relevance Scoring
- **Query Embedding**: Combines persona and job description into a unified query representation
- **Section Embeddings**: Generates semantic embeddings for all document sections using a lightweight transformer model
- **Similarity Computation**: Calculates cosine similarity between the query and each section to identify semantic relevance
- **Top-K Selection**: Ranks all sections and selects the 5 most relevant based on similarity scores

### Phase 3: Subsection Analysis & JSON Output
- **Content Refinement**: Cleans and summarizes selected sections, ensuring proper sentence structure and readability
- **Structured Output**: Formats results according to the specified JSON schema with metadata, extracted sections, and refined analysis
- **Quality Assurance**: Validates output format and content quality before final export

## 📚 Tools & Libraries

**Core Dependencies:**
- **PyMuPDF (fitz)**: Lightweight, fast PDF text extraction (~15MB)
- **Sentence-Transformers**: Semantic embedding generation using `paraphrase-MiniLM-L6-v2` (~80MB)
- **NumPy**: Efficient vector operations and similarity calculations
- **scikit-learn**: Cosine similarity computation and metrics
- **PyTorch**: CPU-optimized backend for transformer models (forced CPU-only mode)

**Supporting Libraries:**
- **Python Standard Library**: JSON handling, regex processing, datetime utilities
- **Pathlib**: Cross-platform file path management

## 🎯 Constraints Handling

The architecture specifically addresses all hackathon constraints:

**CPU-Only Operation**: Forces PyTorch to CPU mode with `torch.set_num_threads(4)` and `CUDA_VISIBLE_DEVICES=''`, ensuring no GPU dependencies.

**Model Size < 1GB**: Uses the compact `paraphrase-MiniLM-L6-v2` model (~80MB) plus dependencies, totaling ~200MB—well under the 1GB limit.

**Runtime < 60 seconds**: Optimized with batch processing (32 sections), efficient chunking strategies, and streamlined text processing. Target runtime: 35-50 seconds.

**Fully Offline**: No external API calls—all processing happens locally with pre-downloaded models and libraries.

## ✅ Why This Works

This architecture excels because it combines **semantic understanding** with **computational efficiency**. Unlike keyword-based approaches, semantic embeddings capture contextual meaning, enabling the system to identify relevant sections even when exact terms don't match.

The modular design ensures **scalability**—new personas or document types require no architectural changes, only different input configurations. The intelligent sectioning preserves document structure while optimizing for semantic analysis.

Performance optimizations (batch processing, sentence-boundary chunking, CPU-specific tuning) ensure the system meets strict time constraints while maintaining accuracy. The lightweight model choice balances semantic capability with resource efficiency, making the solution both effective and deployable in constrained environments.

This approach is **domain-agnostic** and can easily adapt to different industries (legal documents, technical manuals, research papers) by simply changing the input persona and job description, demonstrating its versatility for real-world applications.
