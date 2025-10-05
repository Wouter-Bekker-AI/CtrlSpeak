"""
Simple PDF Parsing with Docling - Quick Start
==============================================

This script demonstrates the most basic usage of Docling:
converting a PDF to Markdown.

Why Docling?
- Handles complex PDFs with tables, images, and multi-column layouts
- No need for custom OCR implementation
- Preserves document structure and formatting
- Works out-of-the-box without configuration

Usage:
    python 01_simple_pdf.py
"""

from docling.document_converter import DocumentConverter

def main():
    # Path to PDF document
    pdf_path = "../documents/technical-architecture-guide.pdf"

    print("=" * 60)
    print("Converting PDF to Markdown with Docling")
    print("=" * 60)
    print(f"Input: {pdf_path}\n")

    # Initialize converter
    converter = DocumentConverter()

    # Convert PDF
    print("Processing PDF...")
    result = converter.convert(pdf_path)

    # Export to Markdown
    markdown = result.document.export_to_markdown()

    # Display results
    print("\n" + "=" * 60)
    print("MARKDOWN OUTPUT")
    print("=" * 60)
    print(markdown[:1000])  # Show first 1000 characters
    print("\n... (truncated for display)")

    # Save to file
    output_path = "output/output_simple.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)

    print(f"\n✓ Full markdown saved to: {output_path}")
    print(f"✓ Total length: {len(markdown)} characters")

if __name__ == "__main__":
    main()
