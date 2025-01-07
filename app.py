import streamlit as st
import os
from rag_system import create_rag_system, query_rag, download_model_if_needed

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the data directory"""
    if not os.path.exists("./data"):
        os.makedirs("./data")
    
    file_path = os.path.join("./data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def main():
    st.title("RAG System Interface")
    
    # Initialize session state for tracking document uploads
    if 'has_documents' not in st.session_state:
        st.session_state.has_documents = False
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents", 
            accept_multiple_files=True,
            type=['txt', 'pdf', 'docx']
        )
        
        if uploaded_files:
            for file in uploaded_files:
                save_uploaded_file(file)
                st.success(f"Uploaded: {file.name}")
                st.session_state.has_documents = True
        
        # Show warning if no documents are uploaded
        if not st.session_state.has_documents:
            st.warning("Please upload some documents before initializing the system.")
        
        if st.button("Initialize/Refresh RAG System", disabled=not st.session_state.has_documents):
            with st.spinner("Initializing RAG system..."):
                try:
                    # Clear existing vector store
                    if os.path.exists("./chroma_db"):
                        import shutil
                        shutil.rmtree("./chroma_db")
                    
                    st.session_state.qa_system = create_rag_system()
                    st.success("RAG system initialized!")
                except Exception as e:
                    st.error(f"Error initializing RAG system: {str(e)}")
                    st.info("Try uploading some documents first if you haven't already.")

    # Show document status
    if not st.session_state.has_documents:
        st.info("👆 Please upload some documents using the sidebar to get started.")
        return

    # Main area for queries
    st.header("Ask Questions")
    user_question = st.text_input("Enter your question:")
    
    if st.button("Ask", disabled=not st.session_state.get('qa_system')):
        if not user_question:
            st.warning("Please enter a question!")
            return
            
        with st.spinner("Thinking..."):
            result = query_rag(st.session_state.qa_system, user_question)
            
            if result["status"] == "success":
                st.write("### Answer:")
                st.write(result["answer"])
                
                st.write("### Sources:")
                for idx, source in enumerate(result["sources"], 1):
                    with st.expander(f"Source {idx}"):
                        st.write(f"**Content:** {source['content']}")
                        st.write(f"**Source:** {source['source']}")
            else:
                st.error(f"Error: {result['message']}")

if __name__ == "__main__":
    main() 