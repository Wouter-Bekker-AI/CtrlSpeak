"""
Main ingestion script for processing markdown documents into ChromaDB.
"""

import os
import asyncio
import logging
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import argparse

import chromadb
import whisper

from .chunker import ChunkingConfig, create_chunker, DocumentChunk
from .embedder import create_embedder

# Import utilities
try:
    from ..utils.db_utils import get_client
    from ..utils.models import IngestionConfig, IngestionResult
except ImportError:
    # For direct execution or testing
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.db_utils import get_client
    from utils.models import IngestionConfig, IngestionResult

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into ChromaDB."""

    def __init__(
        self,
        config: IngestionConfig,
        documents_folder: str = "documents",
        clean_before_ingest: bool = True
    ):
        """
        Initialize ingestion pipeline.

        Args:
            config: Ingestion configuration
            documents_folder: Folder containing markdown documents
            clean_before_ingest: Whether to clean existing data before ingestion (default: True)
        """
        self.config = config
        self.documents_folder = documents_folder
        self.clean_before_ingest = clean_before_ingest

        # Initialize components
        self.chunker_config = ChunkingConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            use_semantic_splitting=config.use_semantic_chunking
        )

        self.chunker = create_chunker(self.chunker_config)
        self.embedder = create_embedder()
        self.chroma_client = get_client()
        self.whisper_model = whisper.load_model("base.en")

        self._initialized = False

    async def initialize(self):
        """Initialize database connections."""
        if self._initialized:
            return

        logger.info("Initializing ingestion pipeline...")
        self._initialized = True
        logger.info("Ingestion pipeline initialized")

    async def close(self):
        """Close database connections."""
        pass

    async def ingest_documents(
        self,
        progress_callback: Optional[callable] = None
    ) -> List[IngestionResult]:
        """
        Ingest all documents from the documents folder.

        Args:
            progress_callback: Optional callback for progress updates

        Returns:
            List of ingestion results
        """
        if not self._initialized:
            await self.initialize()

        collection_name = "rag_collection"
        if self.clean_before_ingest:
            logger.warning(f"Deleting existing collection: {collection_name}")
            try:
                self.chroma_client.delete_collection(name=collection_name)
            except Exception as e:
                logger.info(f"Collection {collection_name} does not exist, skipping deletion.")

        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        document_files = self._find_document_files()

        if not document_files:
            logger.warning(f"No supported document files found in {self.documents_folder}")
            return []

        logger.info(f"Found {len(document_files)} document files to process")

        results = []

        for i, file_path in enumerate(document_files):
            try:
                logger.info(f"Processing file {i+1}/{len(document_files)}: {file_path}")

                result = await self._ingest_single_document(file_path, collection)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(document_files))

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(IngestionResult(
                    document_id="",
                    title=os.path.basename(file_path),
                    chunks_created=0,
                    errors=[str(e)]
                ))

        total_chunks = sum(r.chunks_created for r in results)
        total_errors = sum(len(r.errors) for r in results)

        logger.info(f"Ingestion complete: {len(results)} documents, {total_chunks} chunks, {total_errors} errors")

        return results

    async def _ingest_single_document(self, file_path: str, collection) -> IngestionResult:
        """
        Ingest a single document.

        Args:
            file_path: Path to the document file
            collection: ChromaDB collection

        Returns:
            Ingestion result
        """
        start_time = datetime.now()

        document_content, docling_doc = self._read_document(file_path)
        document_title = self._extract_title(document_content, file_path)
        document_source = os.path.abspath(file_path)

        document_metadata = self._extract_document_metadata(document_content, file_path)

        logger.info(f"Processing document: {document_title}")

        chunks = await self.chunker.chunk_document(
            content=document_content,
            title=document_title,
            source=document_source,
            metadata=document_metadata,
            docling_doc=docling_doc
        )

        if not chunks:
            logger.warning(f"No chunks created for {document_title}")
            return IngestionResult(
                document_id="",
                title=document_title,
                chunks_created=0,
                errors=["No chunks created"]
            )

        logger.info(f"Created {len(chunks)} chunks")

        embedded_chunks = self.embedder.embed_chunks(chunks)
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")

        self._save_to_chromadb(embedded_chunks, document_source, collection)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return IngestionResult(
            document_id=document_source,
            title=document_title,
            chunks_created=len(chunks),
            processing_time_ms=processing_time,
            errors=[]
        )

    def _find_document_files(self) -> List[str]:
        """Find all supported document files in the documents folder."""
        if not os.path.exists(self.documents_folder):
            logger.error(f"Documents folder not found: {self.documents_folder}")
            return []

        patterns = [
            "*.md", "*.markdown", "*.txt",
            "*.pdf",
            "*.docx", "*.doc",
            "*.pptx", "*.ppt",
            "*.xlsx", "*.xls",
            "*.html", "*.htm",
            "*.mp3",
        ]
        files = []

        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.documents_folder, "**", pattern), recursive=True))

        return sorted(files)

    def _read_document(self, file_path: str) -> tuple[str, Optional[Any]]:
        """
        Read document content from file - supports multiple formats via Docling.

        Returns:
            Tuple of (markdown_content, docling_document)
        """
        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext == '.mp3':
            content = self._transcribe_audio(file_path)
            return (content, None)

        try:
            from docling.document_converter import DocumentConverter

            logger.info(f"Converting {file_ext} file using Docling: {os.path.basename(file_path)}")

            converter = DocumentConverter()
            result = converter.convert(file_path)

            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully converted {os.path.basename(file_path)} to markdown")

            return (markdown_content, result.document)

        except Exception as e:
            logger.error(f"Failed to convert {file_path} with Docling: {e}")
            logger.warning(f"Falling back to raw text extraction for {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return (f.read(), None)
            except:
                return (f"[Error: Could not read file {os.path.basename(file_path)}]", None)

    def _transcribe_audio(self, file_path: str) -> str:
        """Transcribe audio file using local Whisper model."""
        try:
            logger.info(f"Transcribing audio file using local Whisper model: {os.path.basename(file_path)}")
            result = self.whisper_model.transcribe(file_path)
            logger.info(f"Successfully transcribed {os.path.basename(file_path)}")
            return result["text"]
        except Exception as e:
            logger.error(f"Failed to transcribe {file_path} with local Whisper: {e}")
            return f"[Error: Could not transcribe audio file {os.path.basename(file_path)} ]"

    def _extract_title(self, content: str, file_path: str) -> str:
        """Extract title from document content or filename."""
        lines = content.split('\n')
        for line in lines[:10]:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        return os.path.splitext(os.path.basename(file_path))[0]

    def _extract_document_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract metadata from document content."""
        return {
            "file_path": file_path,
            "file_size": len(content),
            "ingestion_date": datetime.now().isoformat(),
            "line_count": len(content.split('\n')),
            "word_count": len(content.split()),
        }

    def _save_to_chromadb(self, chunks: List[DocumentChunk], source: str, collection):
        """Save document and chunks to ChromaDB."""
        if not chunks:
            return

        ids = [f"{source}_{chunk.index}" for chunk in chunks]
        contents = [chunk.content for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )

async def main():
    """Main function for running ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--documents", "-d", default="documents", help="Documents folder path")
    parser.add_argument("--no-clean", action="store_true", help="Skip cleaning existing data before ingestion")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for splitting documents")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap size")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic chunking")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    config = IngestionConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        use_semantic_chunking=not args.no_semantic
    )

    pipeline = DocumentIngestionPipeline(
        config=config,
        documents_folder=args.documents,
        clean_before_ingest=not args.no_clean
    )

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total} documents processed")

    try:
        start_time = datetime.now()
        results = await pipeline.ingest_documents(progress_callback)
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        print("\n" + "="*50)
        print("INGESTION SUMMARY")
        print("="*50)
        print(f"Documents processed: {len(results)}")
        print(f"Total chunks created: {sum(r.chunks_created for r in results)}")
        print(f"Total errors: {sum(len(r.errors) for r in results)}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print()

        for result in results:
            status = "Success" if not result.errors else "Fail"
            print(f"{status} {result.title}: {result.chunks_created} chunks")

            if result.errors:
                for error in result.errors:
                    print(f"  Error: {error}")

    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise
    finally:
        await pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
