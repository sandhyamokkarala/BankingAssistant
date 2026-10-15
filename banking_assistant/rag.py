import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
from pypdf.errors import PdfReadError

from banking_assistant.emi import calculate_emi


NO_ANSWER = "I don't have enough information in the bank documents to answer this question."
NO_READABLE_TEXT = (
    "No readable text was extracted from any uploaded PDF. "
    "Check that the PDFs contain selectable text and are not image-only scans."
)

ANSWER_PROMPT = """You are a careful banking customer-support assistant.
Answer policy and product questions using only the bank-document context below.
Do not invent rates, fees, eligibility rules, timelines, or required documents.
If the context does not contain the answer, say exactly:
"I don't have enough information in the bank documents to answer this question."
Use the conversation only to resolve references such as "it" or "that loan"; it is
not a source of banking facts. Cite supporting passages with their source labels,
for example [Source 1]. Do not claim to access customer accounts or determine
personal loan eligibility.

You may use the calculate_emi tool only when the customer explicitly requests an
EMI calculation and provides principal, annual interest rate (as a percentage),
and tenure (in months). The tool calculates arithmetic only; do not present its
result as an offer or quote. For loan conditions, rely on the bank documents.

Bank-document context:
{context}"""

CONTEXTUALIZE_PROMPT = """Rewrite the latest customer question as a standalone search query.
Use the recent conversation only to resolve references and keep the meaning unchanged.
Return only the search query; do not answer it.

Recent conversation:
{history}

Latest question:
{question}"""


@tool
def calculate_emi_tool(principal: float, annual_interest_rate: float, tenure_months: int) -> str:
    """Calculate monthly EMI from principal, annual interest rate in percent, and tenure in months."""
    amount = calculate_emi(principal, annual_interest_rate, tenure_months)
    return f"Estimated monthly installment: INR {amount:,.2f}. This is an arithmetic estimate, not a bank offer."


def resolve_api_key(api_key: str | None = None) -> str:
    key = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise ValueError("Enter an OpenAI API key in the sidebar or configure OPENAI_API_KEY.")
    return key


def _load_pdf_documents(path: Path, source: str) -> list[Document]:
    page_text: dict[int, str] = {}
    try:
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf):
                text = page.get_text("text", sort=True).strip()
                if text:
                    page_text[page_number] = text
    except (OSError, ValueError, RuntimeError, pymupdf.FileDataError):
        pass

    try:
        for page_number, document in enumerate(PyPDFLoader(str(path)).load()):
            text = document.page_content.strip()
            if text and page_number not in page_text:
                page_text[page_number] = text
    except (OSError, ValueError, RuntimeError, PdfReadError):
        pass

    return [
        Document(
            page_content=text,
            metadata={"source": source, "page": page_number},
        )
        for page_number, text in sorted(page_text.items())
    ]


def build_knowledge_base(
    uploaded_files: list[Any] | None,
    api_key: str | None = None,
) -> tuple[FAISS, int]:
    if not uploaded_files:
        raise ValueError("Upload at least one bank policy PDF before indexing.")

    for uploaded_file in uploaded_files:
        if Path(uploaded_file.name).suffix.lower() != ".pdf":
            raise ValueError(f"{Path(uploaded_file.name).name} is not a PDF.")

    key = resolve_api_key(api_key)
    documents = []
    with tempfile.TemporaryDirectory(prefix="banking-assistant-") as temp_dir:
        for index, uploaded_file in enumerate(uploaded_files):
            safe_name = Path(uploaded_file.name).name
            path = Path(temp_dir) / f"{index}-{safe_name}"
            path.write_bytes(uploaded_file.getvalue())
            documents.extend(_load_pdf_documents(path, safe_name))

    if not documents:
        raise ValueError(NO_READABLE_TEXT)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError(NO_READABLE_TEXT)

    embeddings = OpenAIEmbeddings(api_key=key)
    return FAISS.from_documents(chunks, embeddings), len(chunks)


def _history_text(history: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"{'Customer' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
        for turn in history[-6:]
    )


def _standalone_query(question: str, history: list[dict[str, Any]], llm: ChatOpenAI) -> str:
    if not history:
        return question
    prompt = ChatPromptTemplate.from_template(CONTEXTUALIZE_PROMPT)
    result = (prompt | llm).invoke({"history": _history_text(history), "question": question})
    return str(result.content).strip() or question


def answer_question(
    question: str,
    history: list[dict[str, Any]],
    vectorstore: FAISS,
    api_key: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    key = resolve_api_key(api_key)
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=key)
    search_query = _standalone_query(question, history, llm)
    documents = vectorstore.similarity_search(search_query, k=4)

    sources: list[dict[str, Any]] = []
    context_parts = []
    for index, document in enumerate(documents, start=1):
        source = Path(str(document.metadata.get("source", "uploaded document"))).name
        page = document.metadata.get("page")
        page_label = f", page {int(page) + 1}" if page is not None else ""
        label = f"Source {index}: {source}{page_label}"
        sources.append({"label": label, "source": source, "page": int(page) + 1 if page is not None else None})
        context_parts.append(f"[{label}]\n{document.page_content}")

    context = "\n\n".join(context_parts) or "No relevant bank-document passages were retrieved."
    messages = [
        SystemMessage(content=ANSWER_PROMPT.format(context=context)),
        *[
            HumanMessage(content=turn["content"]) if turn["role"] == "user"
            else AIMessage(content=turn["content"])
            for turn in history[-6:]
        ],
        HumanMessage(content=question),
    ]

    assistant = llm.bind_tools([calculate_emi_tool]).invoke(messages)
    for _ in range(3):
        tool_calls = getattr(assistant, "tool_calls", [])
        if not tool_calls:
            break
        messages.append(assistant)
        for tool_call in tool_calls:
            if tool_call["name"] != calculate_emi_tool.name:
                raise RuntimeError(f"Unexpected tool requested by model: {tool_call['name']}")
            result = calculate_emi_tool.invoke(tool_call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        assistant = llm.bind_tools([calculate_emi_tool]).invoke(messages)
    else:
        if getattr(assistant, "tool_calls", []):
            raise RuntimeError("The assistant exceeded the allowed number of tool calls.")

    answer = assistant.content
    if not isinstance(answer, str):
        answer = "\n".join(
            block["text"] for block in answer
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return answer.strip() or NO_ANSWER, sources
