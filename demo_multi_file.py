"""
Example: Multi-File Upload Demo
=================================

This script demonstrates how to use the RAGPipeline's multi-file upload feature
to index and query across multiple PDF documents simultaneously.

Before running:
1. Ensure you have setup your MODEL_PATH in .env
2. Place some PDF files in a folder
3. Update the pdf_files list below with your file paths
"""

import os
from rag_pipeline import RAGPipeline


def progress_callback(current, total, filename):
    """Callback function to track file loading progress."""
    print(f"  [{current}/{total}] Loading: {filename}")


def main():
    print("=" * 70)
    print("RAG Pipeline - Multi-File Upload Demo")
    print("=" * 70)
    
    # Initialize the pipeline
    print("\n1. Initializing RAGPipeline...")
    rag = RAGPipeline(
        persist_dir="./storage",  # Enable index persistence
        n_gpu_layers=-1           # Use GPU if available
    )
    print("   ✓ Pipeline initialized")
    
    # Example: Replace these with your actual PDF file paths
    pdf_files = [
        "example_doc1.pdf",
        "example_doc2.pdf",
        "example_doc3.pdf",
    ]
    
    # Check if files exist (for demo purposes)
    existing_files = [f for f in pdf_files if os.path.exists(f)]
    
    if not existing_files:
        print("\n⚠️  No PDF files found. Please update the pdf_files list with actual file paths.")
        print("   Example files to use:")
        print("   - certificate_of_analysis.pdf")
        print("   - material_specification.pdf")
        print("   - bse_tse_declaration.pdf")
        return
    
    print(f"\n2. Found {len(existing_files)} PDF(s) to process:")
    for f in existing_files:
        print(f"   - {f}")
    
    # Build the index from multiple files
    print("\n3. Building unified index from all PDFs...")
    print("   (This may take a few minutes depending on file sizes)")
    
    try:
        rag.build_from_multiple_pdfs(
            existing_files,
            classify_docs=True,  # Enable pharmaceutical document classification
            progress_callback=progress_callback
        )
        print("   ✓ Index built successfully!")
    except Exception as e:
        print(f"   ✗ Error building index: {e}")
        return
    
    # Display statistics
    print("\n4. Index Statistics:")
    stats = rag.get_stats()
    print(f"   - Total Files:  {stats['total_files']}")
    print(f"   - Total Pages:  {stats['total_pages']}")
    print(f"   - Total Chunks: {stats['total_chunks']}")
    
    if stats.get('file_names'):
        print("\n   Indexed files:")
        for fname in stats['file_names']:
            print(f"     • {fname}")
    
    if stats['classified'] and stats.get('doc_type_counts'):
        print("\n   Document type distribution:")
        for doc_type, count in sorted(stats['doc_type_counts'].items()):
            print(f"     • {doc_type}: {count} pages")
    
    # Example queries
    print("\n5. Example Queries:")
    print("-" * 70)
    
    example_queries = [
        "What are the storage conditions?",
        "Is there a BSE/TSE declaration?",
        "What is the batch number?",
        "Who is the manufacturer?",
    ]
    
    for query in example_queries:
        print(f"\n📋 Query: {query}")
        print("   " + "-" * 65)
        
        try:
            result = rag.query_with_sources(query, classify=True)
            
            # Print answer
            print(f"   Answer: {result['answer'][:200]}...")
            
            # Print sources
            if result['sources']:
                print(f"\n   Sources ({len(result['sources'])} chunks):")
                for i, source in enumerate(result['sources'][:3], 1):  # Show first 3
                    confidence = f"{source['score']:.1f}%" if source['score'] else "N/A"
                    print(f"     {i}. {source['file']} (page {source['page']}) - {confidence} confidence")
            
            if result.get('query_category'):
                print(f"   Classified as: {result['query_category']}")
                
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
