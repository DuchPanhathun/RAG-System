from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
import logging
from huggingface_hub import hf_hub_download
import os

def get_local_model_path():
    """Get the path to the local GGUF model"""
    return "/Users/thun/Desktop/RAG/Llama-3.2-3B-Instruct-Q8_0-GGUF/llama-3.2-3b-instruct-q8_0.gguf"

def download_model_if_needed():
    """Check if local model exists"""
    model_path = get_local_model_path()
    if not os.path.exists(model_path):
        raise Exception(f"Model not found at {model_path}")
    return model_path

def create_rag_system():
    # 1. Initialize the LLM
    try:
        model_path = get_local_model_path()
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.7,
            max_tokens=2000,
            n_ctx=2048,
            top_p=0.95,
            n_gpu_layers=32,
            verbose=True,
        )
    except Exception as e:
        raise Exception(f"Error initializing LLM: {str(e)}")

    # 2. Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    except Exception as e:
        raise Exception(f"Error initializing embeddings: {str(e)}")

    # 3. Create document loader and text splitter
    try:
        if not os.path.exists("./data"):
            os.makedirs("./data")
            with open("./data/sample.txt", "w") as f:
                f.write("This is a sample document. Please upload your own documents to get started.")

        # Create separate loaders for different file types
        pdf_loader = DirectoryLoader(
            "./data",
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        txt_loader = DirectoryLoader(
            "./data",
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        
        # Load documents from both loaders
        pdf_documents = pdf_loader.load()
        txt_documents = txt_loader.load()
        
        # Combine all documents
        documents = pdf_documents + txt_documents
        
        if not documents:
            raise Exception("No documents found in ./data directory. Please upload some documents first.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise Exception("No content found in documents after splitting.")

        print(f"Loaded {len(documents)} documents and split into {len(splits)} chunks")

    except Exception as e:
        raise Exception(f"Error processing documents: {str(e)}")

    # 4. Create FAISS vector store instead of Chroma
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # Save the FAISS index locally
    vectorstore.save_local("faiss_index")

    # 5. Create RAG chain
    try:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        raise Exception(f"Error creating QA chain: {str(e)}")

# Usage example
def query_rag(qa_chain, question: str):
    try:
        response = qa_chain({"query": question})
        sources = [{"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")} 
                  for doc in response["source_documents"]]
        
        return {
            "status": "success",
            "answer": response["result"],
            "sources": sources
        }
    except Exception as e:
        logging.error(f"Error during RAG query: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "answer": None,
            "sources": None
        }

# Initialize and use
if __name__ == "__main__":
    qa_system = create_rag_system()
    
    # Example query
    result = query_rag(qa_system, "What is the capital of France?")
    print("Answer:", result["answer"])
    print("\nSources used:")
    for idx, source in enumerate(result["sources"], 1):
        print(f"\nSource {idx}:")
        print(source)