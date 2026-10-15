# AI Banking Assistant

A browser-based Streamlit training app with a polished Bankwise banking-assistant interface. It demonstrates PDF ingestion, LangChain text splitting, OpenAI embeddings, FAISS retrieval, conversational retrieval, source references, and an EMI calculation tool.

## Setup on Windows

```powershell
Set-Location "C:\Users\ss\OneDrive\Documents\Python Programming\BankingAssistant"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Put your own key in the `OPENAI_API_KEY` value in `.env`, save the file, then start the app from the activated environment:

```powershell
Set-Location "C:\Users\ss\OneDrive\Documents\Python Programming\BankingAssistant"
.\.venv\Scripts\Activate.ps1
streamlit run streamlit_app.py
```

Alternatively, set `$env:OPENAI_API_KEY` in the PowerShell session before launching Streamlit. The application reads the key from the process environment or the project's `.env` file; the blank sidebar override is optional and does not display the configured environment key.

Upload official policy PDFs in the sidebar, select **Index documents**, and ask a question in the chat. Indexing extracts text page by page with PyMuPDF and falls back to PyPDF when a page has no extractable text, skips blank pages, splits the remaining text, creates OpenAI embeddings, and builds an in-memory FAISS vector store. The index is rebuilt when you index files; uploaded PDFs and the vector index are not persisted by the app. Follow-up questions use recent conversation turns to clarify the retrieval query. Answers show the source PDF and page for the retrieved passages. Missing keys, missing PDFs, and the case where none of the uploaded PDFs yields readable text are shown as actionable UI messages.

The assistant is instructed to answer policy questions only from retrieved context and to abstain when the documents do not contain an answer. It does not determine personal loan eligibility or access accounts. The EMI calculator and LangChain tool perform arithmetic only and do not provide a bank quote.

## Privacy and deployment

This is an educational demo, not a production banking service. Document text is sent to the configured OpenAI API for embeddings and inference. Use only public or bank-approved documents and never upload customer records, credentials, or other sensitive data. Configure approved providers, authentication, access controls, retention, audit logging, and bank security/compliance review before any real deployment. Do not commit `.env` or API keys.

## Tests

```powershell
Set-Location "C:\Users\ss\OneDrive\Documents\Python Programming\BankingAssistant"
.\.venv\Scripts\python.exe -m pytest -q
```

The RAG tests use a generated local PDF and mocked OpenAI/FAISS calls. They verify document loading, environment-key resolution, embedding/index construction, retrieval, and source labels without making a live OpenAI request. To test a real RAG answer, provide your own API key and official PDF, start the app, index the PDF, and ask a question.
