# 📄 Chat with PDFs Locally using Ollama & LangChain

Interact with your PDF documents using local embeddings and LLMs. This app uses [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss), and [Ollama](https://ollama.com/) to let you ask natural language questions about the contents of your uploaded PDFs — all locally, with no cloud or OpenAI dependency.

![Streamlit App Screenshot](preview.png) <!-- Optional: Replace with an actual screenshot -->

---

## 🚀 Features

- 📤 Upload and process multiple PDF files
- 🧠 Create and manage a FAISS vector store locally
- 💬 Ask natural language questions about your PDFs
- 🧾 View previous questions and answers
- ⚙️ Full local setup (no API keys or cloud calls required)
- 🪄 Powered by Ollama + DeepSeek LLM + LangChain

---

## 🧩 Technologies Used

- [Streamlit](https://streamlit.io/) — UI Framework
- [LangChain](https://www.langchain.com/) — LLM and QA Chain
- [FAISS](https://github.com/facebookresearch/faiss) — Local vector store
- [Ollama](https://ollama.com/) — Local LLM and embeddings
- [PyPDFLoader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf) — PDF content loader

---

## 🖥️ Installation

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/pdf-chat-local.git
cd pdf-chat-local
# RAG_Chat

    Create a Virtual Environment (optional but recommended)

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

    Install Dependencies

pip install -r requirements.txt

    Install and Run Ollama

    You must have Ollama installed and a compatible model (e.g., deepseek-r1:1.5b) pulled locally.

ollama pull deepseek-r1:1.5b

    Run the App

streamlit run app.py

📁 Project Structure

├── app.py                  # Main Streamlit application
├── data/                   # Uploaded PDFs will be stored here
├── faiss_index/            # Vector index folder (auto-created)
├── requirements.txt
└── README.md

🧠 Example Usage

    Upload your PDF files using the file uploader.

    Click "⚙️ Process Documents" to generate embeddings and build the vector store.

    Ask questions in natural language — e.g., "What is the main topic of the document?" or "Summarize section 3."

    View previous questions and answers.

⚠️ Notes

    All operations are local — no data is sent to any external APIs.

    This app assumes deepseek-r1:1.5b is available locally via Ollama.

    FAISS index and uploaded files persist unless cleared via the UI.

📜 License

MIT License. See LICENSE for more information.
🙌 Acknowledgements

    LangChain

    Ollama

    Streamlit

✨ Future Enhancements

    PDF text preview and search

    Dark mode toggle

    Model selection dropdown

    Export chat history


---

### ✅ Bonus: `requirements.txt`

If you haven’t already, include this:

```txt
streamlit
langchain
faiss-cpu
langchain-community
langchain-ollama
pypdf
