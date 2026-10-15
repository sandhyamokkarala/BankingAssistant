from __future__ import annotations

from collections.abc import Iterator
from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import banking_assistant.rag as rag


class UploadedFile:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def make_text_pdf(text: str | list[str]) -> bytes:
    page_texts = [text] if isinstance(text, str) else text
    page_objects = [3 + page_index * 2 for page_index in range(len(page_texts))]
    font_object = 3 + len(page_texts) * 2
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (
            f"<< /Type /Pages /Kids [{' '.join(f'{page} 0 R' for page in page_objects)}] "
            f"/Count {len(page_texts)} >>"
        ).encode(),
    ]
    for page_index, text_value in enumerate(page_texts):
        page_object = page_objects[page_index]
        content_object = page_object + 1
        content_stream = f"BT /F1 12 Tf 72 720 Td ({text_value}) Tj ET".encode("ascii")
        objects.extend([
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 {font_object} 0 R >> >> "
                f"/Contents {content_object} 0 R >>"
            ).encode(),
            b"<< /Length " + str(len(content_stream)).encode() + b" >>\nstream\n"
            + content_stream + b"\nendstream",
        ])
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    output = BytesIO()
    output.write(b"%PDF-1.4\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(output.tell())
        output.write(f"{object_number} 0 obj\n".encode())
        output.write(body)
        output.write(b"\nendobj\n")
    xref_offset = output.tell()
    output.write(f"xref\n0 {len(offsets)}\n".encode())
    output.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.write(f"{offset:010d} 00000 n \n".encode())
    output.write(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return output.getvalue()


def test_resolves_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-environment")
    assert rag.resolve_api_key() == "test-key-from-environment"


def test_missing_api_key_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        rag.resolve_api_key()


def test_missing_pdfs_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="Upload at least one"):
        rag.build_knowledge_base([])


def test_builds_faiss_index_from_pdf_and_answers_with_retrieved_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-environment")
    embedding_keys: list[str] = []
    indexed_documents: list[Document] = []

    class FakeEmbeddings:
        def __init__(self, *, api_key: str) -> None:
            embedding_keys.append(api_key)

    class FakeVectorStore:
        def similarity_search(self, query: str, k: int) -> list[Document]:
            assert query == "What documents are needed for a home loan?"
            assert k == 4
            return [
                Document(
                    page_content="Home loan applicants must provide identity proof.",
                    metadata={"source": "home_loan.pdf", "page": 0},
                )
            ]

    def fake_from_documents(documents: list[Document], embeddings: FakeEmbeddings) -> FakeVectorStore:
        assert isinstance(embeddings, FakeEmbeddings)
        indexed_documents.extend(documents)
        return FakeVectorStore()

    invocations: list[list[object]] = []

    class FakeChatModel:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["api_key"] == "test-key-from-environment"

        def bind_tools(self, tools: list[object]) -> FakeChatModel:
            assert [getattr(tool, "name", None) for tool in tools] == ["calculate_emi_tool"]
            return self

        def invoke(self, messages: list[object]) -> AIMessage:
            invocations.append(messages)
            return AIMessage(content="Identity proof is required. [Source 1]")

    monkeypatch.setattr(rag, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag.FAISS, "from_documents", fake_from_documents)
    monkeypatch.setattr(rag, "ChatOpenAI", FakeChatModel)

    vectorstore, chunk_count = rag.build_knowledge_base(
        [UploadedFile("home_loan.pdf", make_text_pdf("Home loan requires identity proof."))]
    )
    answer, sources = rag.answer_question(
        "What documents are needed for a home loan?",
        [],
        vectorstore,
    )

    assert chunk_count == 1
    assert embedding_keys == ["test-key-from-environment"]
    assert len(indexed_documents) == 1
    assert "Home loan requires identity proof." in indexed_documents[0].page_content
    assert indexed_documents[0].metadata["source"] == "home_loan.pdf"
    assert answer == "Identity proof is required. [Source 1]"
    assert sources == [{"label": "Source 1: home_loan.pdf, page 1", "source": "home_loan.pdf", "page": 1}]
    assert any(
        "Home loan applicants must provide identity proof." in str(message.content)
        for message in invocations[0]
        if hasattr(message, "content")
    )


def test_rejects_invalid_pdf_without_leaking_parser_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="No readable text was extracted from any uploaded PDF") as error:
        rag.build_knowledge_base([UploadedFile("broken.pdf", b"not a PDF")])
    assert "Traceback" not in str(error.value)


def test_indexes_text_from_multiple_pages_and_ignores_blank_pages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-environment")
    indexed_documents: list[Document] = []

    class FakeEmbeddings:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key-from-environment"

    def fake_from_documents(documents: list[Document], embeddings: FakeEmbeddings) -> object:
        indexed_documents.extend(documents)
        return object()

    monkeypatch.setattr(rag, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag.FAISS, "from_documents", fake_from_documents)

    _, chunk_count = rag.build_knowledge_base([
        UploadedFile(
            "multi_page_policy.pdf",
            make_text_pdf([
                "Home loan applicants must provide identity proof.",
                "",
                "Applicants must provide proof of income.",
            ]),
        )
    ])

    assert chunk_count == 2
    assert [document.metadata["page"] for document in indexed_documents] == [0, 2]
    assert [document.metadata["source"] for document in indexed_documents] == [
        "multi_page_policy.pdf",
        "multi_page_policy.pdf",
    ]
    assert all(document.page_content.strip() for document in indexed_documents)


def test_falls_back_to_pypdf_when_pymupdf_returns_blank_pages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class BlankPage:
        def get_text(self, *_args: object, **_kwargs: object) -> str:
            return ""

    class BlankPdf:
        def __enter__(self) -> BlankPdf:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __iter__(self) -> Iterator[BlankPage]:
            return iter([BlankPage()])

    pdf_path = tmp_path / "fallback.pdf"
    pdf_path.write_bytes(make_text_pdf("This selectable text is recovered by pypdf."))
    monkeypatch.setattr(rag.pymupdf, "open", lambda _path: BlankPdf())

    documents = rag._load_pdf_documents(pdf_path, "fallback.pdf")

    assert len(documents) == 1
    assert "This selectable text is recovered by pypdf." in documents[0].page_content
    assert documents[0].metadata == {"source": "fallback.pdf", "page": 0}


def test_indexes_readable_pdf_when_another_uploaded_pdf_is_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    indexed_documents: list[Document] = []

    class FakeEmbeddings:
        def __init__(self, *, api_key: str) -> None:
            assert api_key == "test-key"

    def fake_from_documents(documents: list[Document], _embeddings: FakeEmbeddings) -> object:
        indexed_documents.extend(documents)
        return object()

    monkeypatch.setattr(rag, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(rag.FAISS, "from_documents", fake_from_documents)

    _, chunk_count = rag.build_knowledge_base([
        UploadedFile("unreadable.pdf", b"not a PDF"),
        UploadedFile("readable.pdf", make_text_pdf("This page contains readable policy text.")),
    ])

    assert chunk_count == 1
    assert len(indexed_documents) == 1
    assert indexed_documents[0].metadata["source"] == "readable.pdf"


def test_reports_no_readable_text_only_when_all_uploaded_pdfs_are_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="No readable text was extracted from any uploaded PDF"):
        rag.build_knowledge_base([
            UploadedFile("blank_one.pdf", make_text_pdf(["", "  "])),
            UploadedFile("blank_two.pdf", make_text_pdf([""])),
        ])
