import os
import shutil
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM

# ---- Setup ----
st.set_page_config("📄 Chat with PDF", layout="wide")

embeddings_model = OllamaEmbeddings(model="deepseek-r1:1.5b")

def get_ollama_llm():
    return OllamaLLM(model="deepseek-r1:1.5b")

def load_and_split_documents():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

def create_vector_store(docs):
    db = FAISS.from_documents(docs, embeddings_model)
    db.save_local("faiss_index")

def delete_vector_store():
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")

def vector_store_exists():
    return os.path.exists("faiss_index")

# ---- Prompt ----
prompt_template = """
Human: Use the following context to answer the question with a concise and factual response (250 words max).
If the answer is unknown, say "I don't know."

<context>
{context}
</context>

Question: {question}

Assistant:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_answer(query):
    db = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(
        llm=get_ollama_llm(),
        chain_type="stuff",
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa({"query": query})['result']

# ---- Streamlit App ----
def main():
    st.title("📄 Chat with PDF using Ollama")
    st.markdown("Upload PDF documents, build a local vector store, and query them using a local LLM.")

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("### 📊 System Status")
        if vector_store_exists():
            st.success("✅ Vector store is ready.")
        else:
            st.warning("⚠️ No vector store found.")
        st.markdown("---")
        st.button("🗑️ Delete Vector Store", on_click=delete_vector_store)
        if st.button("🧹 Clear Uploaded Files"):
            if os.path.exists("data"):
                shutil.rmtree("data")
                st.warning("Uploaded files cleared.")

    with col1:
        st.subheader("📤 Upload PDFs")
        uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            os.makedirs("data", exist_ok=True)
            for f in uploaded_files:
                with open(os.path.join("data", f.name), "wb") as out_file:
                    out_file.write(f.getbuffer())
            st.success(f"{len(uploaded_files)} file(s) saved to `/data`")

            with st.expander("📂 View Uploaded Files", expanded=False):
                for f in uploaded_files:
                    st.markdown(f"- 📄 `{f.name}`")

    st.markdown("---")

    # --- Build Vector Store ---
    st.subheader("🧠 Build or Update Vector Store")
    if st.button("⚙️ Process Documents"):
        with st.spinner("Splitting and embedding PDFs..."):
            docs = load_and_split_documents()
            if docs:
                create_vector_store(docs)
                st.success("Vector store built successfully!")
            else:
                st.error("No documents to process.")

    st.markdown("---")

    # --- Ask a Question ---
    st.subheader("💬 Ask a Question")
    st.markdown("Type a question related to your uploaded PDFs.")

    query = st.text_input("Question:")

    if "history" not in st.session_state:
        st.session_state.history = []

    if st.button("🔍 Get Answer") and query:
        if not vector_store_exists():
            st.error("❌ Please build the vector store first.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = get_answer(query)
                    st.session_state.history.append((query, answer))
                    st.success("✅ Answer generated!")
                    st.markdown(f"**🧠 Answer:**\n\n{answer}")
                except Exception as e:
                    st.error(f"⚠️ Error: {e}")

    if st.session_state.history:
        with st.expander("🕘 View Q&A History"):
            for i, (q, a) in enumerate(reversed(st.session_state.history), 1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")
                st.markdown("---")

if __name__ == "__main__":
    main()
