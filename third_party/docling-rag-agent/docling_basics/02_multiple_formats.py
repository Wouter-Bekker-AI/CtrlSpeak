"""
Multi-Format Document Processing with Docling
==============================================

This script demonstrates Docling's ability to handle multiple
document formats with a unified API.

Supported formats:
- PDF (.pdf)
- Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- HTML (.html, .htm)
- Images (.png, .jpg)
- And more...

Usage:
    python 02_multiple_formats.py
"""

from docling.document_converter import DocumentConverter
from pathlib import Path

def process_document(file_path: str, converter: DocumentConverter) -> dict:
    """Process a single document and return metadata."""
    try:
        print(f"\n📄 Processing: {Path(file_path).name}")

        # Convert document
        result = converter.convert(file_path)

        # Export to markdown
        markdown = result.document.export_to_markdown()

        # Get document info
        doc_info = {
            'file': Path(file_path).name,
            'format': Path(file_path).suffix,
            'status': 'Success',
            'markdown_length': len(markdown),
            'preview': markdown[:200].replace('\n', ' ')
        }

        # Save output
        output_file = f"output/output_{Path(file_path).stem}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown)

        doc_info['output_file'] = output_file

        print(f"   ✓ Converted successfully")
        print(f"   ✓ Output: {output_file}")

        return doc_info

    except Exception as e:
        print(f"   ✗ Error: {e}")
        return {
            'file': Path(file_path).name,
            'format': Path(file_path).suffix,
            'status': 'Failed',
            'error': str(e)
        }

def main():
    print("=" * 60)
    print("Multi-Format Document Processing with Docling")
    print("=" * 60)

    # List of documents to process
    documents = [
        "../documents/technical-architecture-guide.pdf",
        "../documents/q4-2024-business-review.pdf",
        "../documents/meeting-notes-2025-01-08.docx",
        "../documents/company-overview.md",
    ]

    # Initialize converter once (reusable)
    converter = DocumentConverter()

    # Process all documents
    results = []
    for doc_path in documents:
        result = process_document(doc_path, converter)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)

    for result in results:
        status_icon = "✓" if result['status'] == 'Success' else "✗"
        print(f"{status_icon} {result['file']} ({result['format']})")
        if result['status'] == 'Success':
            print(f"   Length: {result['markdown_length']} chars")
            print(f"   Preview: {result['preview']}...")
        else:
            print(f"   Error: {result.get('error', 'Unknown')}")
        print()

    success_count = sum(1 for r in results if r['status'] == 'Success')
    print(f"Converted {success_count}/{len(results)} documents successfully")

if __name__ == "__main__":
    main()
