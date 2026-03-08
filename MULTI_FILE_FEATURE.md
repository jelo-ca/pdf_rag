# Multi-File Upload Feature

## Overview

The RAG pipeline now supports uploading and processing multiple PDF files simultaneously through a queue-based system. This allows you to index multiple pharmaceutical documents into a single unified index for cross-document querying.

## Features

### 1. Multi-PDF Indexing

- Upload multiple PDF files at once through the Gradio UI
- Files are processed sequentially and combined into a single searchable index
- Progress tracking shows which files have been loaded
- All documents share the same vector space for optimal retrieval across files

### 2. Queue-Based Processing

- Files are added to a processing queue
- Progress callback provides real-time updates as each file is processed
- Status display shows:
  - Number of files being processed
  - Current file being loaded
  - Processing completion status

### 3. Enhanced Statistics

- View statistics across all indexed files:
  - Total number of files indexed
  - Total pages across all documents
  - Total chunks created
  - List of indexed file names
  - Document type distribution (if classification is enabled)

## Usage

### Gradio UI

1. **Upload Multiple Files**

   - Click the "Upload PDF(s)" button
   - Select one or more PDF files (Ctrl/Cmd+Click for multiple selection)
   - Or drag and drop multiple files into the upload area

2. **Build Pipeline**

   - Optionally enable "Classify document pages by pharma type"
   - Click "Build Pipeline" button
   - Watch the progress updates in the stats panel
   - Wait for "✓ Ready —" status message

3. **Query Across All Documents**
   - Ask questions in the chat interface
   - The system will retrieve relevant information from all indexed PDFs
   - Sources panel shows which document and page each answer came from

### Python API

#### Single File (Original)

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.build("document.pdf", classify_docs=False)
answer = rag.query("What are the storage conditions?")
```

#### Multiple Files (New)

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

# Define a progress callback (optional)
def show_progress(current, total, filename):
    print(f"Loading {current}/{total}: {filename}")

# Build from multiple PDFs
pdf_files = [
    "certificate_of_analysis.pdf",
    "material_specification.pdf",
    "bse_tse_declaration.pdf"
]

rag.build_from_multiple_pdfs(
    pdf_files,
    classify_docs=True,
    progress_callback=show_progress
)

# Query across all documents
result = rag.query_with_sources("What is the BSE/TSE status?")
print(result["answer"])
for source in result["sources"]:
    print(f"- {source['file']}, page {source['page']}")
```

## Technical Details

### RAGPipeline Methods

#### `build_from_multiple_pdfs(pdf_paths, classify_docs, progress_callback)`

Processes multiple PDF files and creates a unified index.

**Parameters:**

- `pdf_paths` (List[str]): List of file paths to PDF documents
- `classify_docs` (bool): Whether to classify pages by pharma document type
- `progress_callback` (Optional[callable]): Function called after each file loads
  - Signature: `callback(current_index: int, total_count: int, filename: str)`

**Behavior:**

- Clears any existing persisted index (multi-document builds are always fresh)
- Loads all PDFs sequentially
- Combines all pages into a single document collection
- Performs classification once on all pages (if enabled)
- Creates a unified chunk and index structure
- Builds hybrid retriever across all documents

#### `get_stats()` (Enhanced)

Returns enhanced statistics for multi-file indexing.

**Returns:**

```python
{
    "total_files": int,        # Number of unique files indexed
    "file_names": List[str],   # List of indexed file names
    "total_pages": int,        # Total pages across all files
    "total_chunks": int,       # Total chunks created
    "doc_type_counts": dict,   # Pharma doc type distribution
    "classified": bool         # Whether classification was performed
}
```

## Benefits

1. **Cross-Document Search**: Query across multiple related documents simultaneously
2. **Unified Context**: Retrieve relevant information regardless of which file it's in
3. **Better Coverage**: Combine related documents (specifications, certificates, etc.) for comprehensive answers
4. **Efficient Processing**: Sequential loading with progress feedback
5. **Metadata Preservation**: Each chunk retains its source file name for proper citation

## Limitations

1. **No Incremental Updates**: Cannot add files to an existing index incrementally

   - Must rebuild the entire index when files change
   - Persisted indexes are cleared for multi-file builds

2. **Memory Usage**: All documents are loaded into memory before chunking

   - May require significant RAM for large document collections
   - Consider processing in batches if memory is limited

3. **Sequential Processing**: Files are processed one at a time
   - Not parallelized to maintain consistent memory usage
   - Large collections may take time to process

## Example Use Cases

### Batch Documentation Processing

```python
# Process an entire batch submission
batch_docs = [
    "batch_record.pdf",
    "certificate_of_analysis.pdf",
    "stability_data.pdf",
    "validation_report.pdf"
]
rag.build_from_multiple_pdfs(batch_docs, classify_docs=True)
rag.query("Is the batch approved for release?")
```

### Supplier Documentation

```python
# Index all supplier qualification documents
supplier_docs = [
    "supplier_audit.pdf",
    "bse_tse_declaration.pdf",
    "allergen_statement.pdf",
    "certificate_of_quality.pdf"
]
rag.build_from_multiple_pdfs(supplier_docs, classify_docs=True)
rag.query("What allergens are present?")
```

### Regulatory Submission

```python
# Combine all submission documents
submission_docs = [
    "cover_letter.pdf",
    "technical_data.pdf",
    "safety_data.pdf",
    "efficacy_data.pdf"
]
rag.build_from_multiple_pdfs(submission_docs)
rag.query("Summarize the safety profile")
```

## Tips

1. **Group Related Documents**: Index related documents together for better context
2. **Use Classification**: Enable `classify_docs=True` for heterogeneous document sets
3. **Monitor Progress**: Use progress callbacks for large batches
4. **Consistent Naming**: Use clear, descriptive filenames for better source citations
5. **Batch Size**: Keep batches under 20-30 documents for optimal performance

## Testing

Run the test suite to verify multi-file functionality:

```bash
pytest tests/test_rag_pipeline.py -v -k "multi"
```

## Future Enhancements

Potential improvements for future versions:

- Incremental index updates (add/remove files without full rebuild)
- Parallel file processing
- Streaming/batch processing for very large collections
- Per-file indexing with cross-file retrieval
- Document versioning and history tracking
