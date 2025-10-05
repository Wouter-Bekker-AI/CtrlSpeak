# Docling Basics - Progressive Tutorial

This folder contains a series of progressive examples demonstrating Docling's capabilities for document processing, from simple PDF conversion to advanced hybrid chunking for RAG systems.

## 📚 What is Docling?

**Docling** is a powerful document processing library that handles complex document formats that are typically challenging for RAG (Retrieval Augmented Generation) systems. Without Docling, you'd need to implement custom OCR, layout analysis, table extraction, and format-specific parsers. Docling handles all of this out-of-the-box.

**Key Advantages:**
- 🔧 **No Custom OCR Required** - Built-in OCR with EasyOCR support
- 📊 **Preserves Structure** - Maintains tables, sections, hierarchies
- 🎯 **Multi-Format Support** - PDF, Word, PowerPoint, Excel, HTML, Audio
- 🤖 **RAG-Ready** - Intelligent chunking optimized for embeddings
- 📝 **Markdown Export** - Clean, consistent output format

## 🎯 Tutorial Progression

### 1️⃣ **Simple PDF Conversion** (`01_simple_pdf.py`)

**What it demonstrates:**
- Most basic Docling usage
- Converting a single PDF to Markdown
- Why Docling is useful for complex PDFs

**Key concepts:**
- `DocumentConverter` - The main entry point
- `export_to_markdown()` - Standard output format

**Run it:**
```bash
python 01_simple_pdf.py
```

**What this covers:**
- How to convert any PDF without configuration
- Docling handles tables, multi-column layouts, and complex formatting automatically
- Clean Markdown output ready for further processing

---

### 2️⃣ **Multiple Document Formats** (`02_multiple_formats.py`)

**What it demonstrates:**
- Processing different file types with the same API
- Batch document conversion
- Unified handling of PDF, Word, PowerPoint, etc.

**Key concepts:**
- Format auto-detection
- Reusable converter instance
- Error handling for multiple documents

**Run it:**
```bash
python 02_multiple_formats.py
```

**What this covers:**
- Docling uses the same API for all document types
- No need for format-specific code
- Process entire document collections with ease

---

### 3️⃣ **Audio Transcription** (`03_audio_transcription.py`)

**What it demonstrates:**
- Audio file transcription with Whisper ASR
- Handling MP3, WAV, M4A, FLAC files
- Timestamp-aware transcripts

**Key concepts:**
- `AsrPipeline` - Audio processing pipeline
- Whisper Turbo model integration
- Timestamp extraction

**Prerequisites:**
FFmpeg must be installed:

**Windows (Chocolatey):**
```bash
choco install ffmpeg
```

**Windows (Conda):**
```bash
conda install -c conda-forge ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
apt-get install ffmpeg  # Debian/Ubuntu
yum install ffmpeg      # RedHat/CentOS
```

**Run it:**
```bash
python 03_audio_transcription.py
```

**What this covers:**
- Convert podcasts, interviews, lectures to searchable text
- Timestamps allow temporal referencing
- Supports 90+ languages automatically

---

### 4️⃣ **Hybrid Chunking** (`04_hybrid_chunking.py`)

**What it demonstrates:**
- Intelligent document chunking for RAG systems
- Token-aware splitting that respects document structure
- Optimization for embedding models

**Key concepts:**
- `HybridChunker` - Structure + token-aware chunking
- Token limits (typical: 512 tokens for embeddings)
- Semantic boundary preservation

**Why Hybrid Chunking?**
- **Problem**: Naive text splitting breaks semantic meaning
- **Solution**: HybridChunker respects paragraphs, sections, tables
- **Benefit**: Better RAG performance with coherent chunks

**Run it:**
```bash
python 04_hybrid_chunking.py
```

**What this covers:**
- How to chunk documents for optimal RAG performance
- Balance between semantic coherence and token limits
- Metadata preservation for context

---

## 🚀 Advanced Features (Optional Enhancements)

Beyond these tutorials, Docling offers additional capabilities for even more robust document processing:

### **Picture Classification & Description**

Add vision-based understanding to your PDFs:

```python
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    granite_picture_description
)
from docling.datamodel.base_models import InputFormat

# Configure picture description for PDFs
pipeline_options = PdfPipelineOptions()
pipeline_options.do_picture_description = True
pipeline_options.picture_description_options = granite_picture_description

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

**Benefits:**
- 🖼️ Automatic image captioning with IBM Granite Vision
- 📊 Diagram and chart descriptions
- 🔍 Makes visual content searchable in RAG systems

### **Code Understanding**

Enhanced processing for technical documents with code:

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_code_enrichment = True  # Enables code syntax understanding
```

**Benefits:**
- 🔤 Syntax highlighting preservation
- 📝 Code block identification
- 🏷️ Language detection

### **Table Structure Recognition**

Advanced table parsing with TableFormer:

```python
from docling.datamodel.pipeline_options import TableFormerMode

pipeline_options = PdfPipelineOptions()
pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
```

**Benefits:**
- 📊 Complex table extraction
- 🔗 Cell relationship preservation
- 📈 Multi-page table handling

---

## 📖 From Basics to Full RAG Agent

These tutorials demonstrate the **building blocks** used in the main RAG agent:

### **Docling Basics** (This Folder)
- Individual scripts showing core concepts
- Focus on understanding each feature
- Quick experimentation and learning

### **Full RAG Agent** (Main Project)
- Complete ingestion pipeline (`ingestion/ingest.py`)
- PostgreSQL + PGVector storage
- Interactive CLI with streaming responses
- Production-ready implementation

**Progression Flow:**
1. **Learn** → Work through `docling_basics/` tutorials
2. **Understand** → See how each piece works independently
3. **Apply** → Explore the full RAG agent implementation
4. **Customize** → Adapt for your specific use case

---

## 🛠️ Installation

All examples require Docling and dependencies:

```bash
# Install base Docling
pip install docling

# For hybrid chunking and ASR (example 3&4)
pip install transformers openai-whisper hf-xet

# OR install everything at once
pip install docling transformers openai-whisper hf-xet
```

---

## 📂 Expected File Structure

The `documents/` folder (one level up) contains example files:
- **PDFs**: `technical-architecture-guide.pdf`, `q4-2024-business-review.pdf`, `client-review-globalfinance.pdf`
- **Word**: `meeting-notes-2025-01-08.docx`, `meeting-notes-2025-01-15.docx`
- **Markdown**: `company-overview.md`, `team-handbook.md`, `mission-and-goals.md`, `implementation-playbook.md`
- **Audio**: `Recording1.mp3`, `Recording2.mp3`, `Recording3.mp3`, `Recording4.mp3`

```
docling-rag-agent/
├── docling_basics/          # This folder - Tutorial scripts
│   ├── 01_simple_pdf.py
│   ├── 02_multiple_formats.py
│   ├── 03_audio_transcription.py
│   ├── 04_hybrid_chunking.py
│   └── README.md
├── documents/               # Source documents (examples provided)
│   ├── technical-architecture-guide.pdf
│   ├── q4-2024-business-review.pdf
│   ├── meeting-notes-2025-01-08.docx
│   ├── company-overview.md
│   ├── Recording1.mp3
│   └── ... (more files)
└── ... (main RAG agent files)
```

---

## 🎓 Learning Path

**Recommended Order:**

1. **Start Here** → `01_simple_pdf.py`
   - Get comfortable with basic conversion
   - See Docling's output format

2. **Expand** → `02_multiple_formats.py`
   - Learn unified API for different formats
   - Understand batch processing

3. **Add Audio** → `03_audio_transcription.py`
   - See how audio fits into document processing
   - Understand ASR pipeline

4. **Optimize for RAG** → `04_hybrid_chunking.py`
   - Critical for production RAG systems
   - Learn about token limits and semantic chunking

5. **Explore Full Agent** → Main project files
   - See everything integrated
   - Production-ready implementation

---

## 💡 Key Takeaways

After completing these tutorials, you'll understand:

✅ **Why Docling?**
- Eliminates need for custom document processing code
- Handles formats that break traditional text extraction
- Provides consistent output across all formats

✅ **When to Use Docling?**
- Building RAG systems with diverse document types
- Processing PDFs with complex layouts
- Need for audio transcription in knowledge bases
- Handling Word, PowerPoint, Excel in automated pipelines

✅ **How Docling Fits RAG?**
- Converts everything to clean Markdown
- HybridChunker optimizes for embedding models
- Preserves structure and metadata for context
- Enables semantic search across all document types

---

## 🔗 Additional Resources

- **Docling Documentation**: https://docling-project.github.io/docling/
- **Docling GitHub**: https://github.com/DS4SD/docling
- **HybridChunker Guide**: https://docling-project.github.io/docling/concepts/chunking/
- **ASR Pipeline**: https://docling-project.github.io/docling/examples/minimal_asr_pipeline/

---

## 🚀 Next Steps

Ready to build your own RAG system? Check out the main project files:
- `ingestion/ingest.py` - Full ingestion pipeline
- `cli.py` - Interactive CLI with streaming
- `rag_agent.py` - RAG tool implementation

These tutorials provide the foundation. The main agent shows the complete picture! 🎯
