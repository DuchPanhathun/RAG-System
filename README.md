# Document Q&A RAG System

A Retrieval-Augmented Generation (RAG) system built with Streamlit that allows users to upload documents and ask questions about their content. The system uses LLaMA 2 for text generation and ChromaDB for vector storage.

## Features

- 📄 Support for multiple document formats (PDF, TXT, DOCX)
- 🔍 Intelligent document chunking and embedding
- 💾 Persistent vector storage using ChromaDB
- 🤖 LLaMA 2 powered question answering
- 🌐 User-friendly Streamlit interface

## Prerequisites

- Python 3.8+
- LLaMA model file (GGUF format)
- At least 16GB RAM recommended
- CUDA-compatible GPU (optional, but recommended)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
2. Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate   # On Windows, use: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt
4. Set up the LLaMA model:
   - Download the LLaMA model in GGUF format
   - Update the model path in `rag_system.py`

## Configuration

Update the model path in `rag_system.py`:

## Usage

1. Start the Streamlit application:
streamlit run app.py```
2. Upload documents:
   - Use the sidebar to upload PDF or TXT files
   - Click "Initialize/Refresh RAG System" after uploading

3. Ask questions:
   - Type your question in the text input
   - Click "Ask" to get answers based on your documents

## Project Structure
.
├── app.py # Streamlit interface
├── rag_system.py # RAG implementation
├── requirements.txt # Project dependencies
├── data/ # Document storage
└── chroma_db/ # Vector store

## Configuration

Key parameters can be adjusted in `rag_system.py`:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `temperature`: LLM creativity (default: 0.7)
- `max_tokens`: Maximum response length (default: 2000)

## Limitations

- Currently supports PDF and TXT files
- Performance depends on the local machine's capabilities
- Requires local storage for document processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

## Acknowledgments

- LangChain for the RAG framework
- Llama for the language model
- Streamlit for the web interface