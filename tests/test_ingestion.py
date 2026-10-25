from garage_rag.ingestion import chunk_pages, clean_text, extract_pdf


def test_extracts_every_page_and_preserves_identity(pdf_path):
    pages, diagnostics = extract_pdf(pdf_path, {"vehicle_models": ["Aurora A1"]})
    assert len(pages) == 4
    assert diagnostics["pages"] == 4
    assert diagnostics["pages_without_text"] == [3]
    assert pages[0].metadata["pdf_page"] == 1
    assert pages[3].metadata["pdf_page"] == 4
    assert pages[0].metadata["document"] == "aurora.pdf"
    assert pages[0].metadata["vehicle_models"] == ["Aurora A1"]
    assert pages[0].metadata["title"] == "Fictional Aurora A1 Owner Guide"
    assert "2.4 bar" in pages[0].page_content


def test_cleaning_preserves_numbers_and_paragraphs():
    assert clean_text("  2.4\u00a0bar\r\n\r\nCheck  tires.\x00 ") == "2.4 bar\n\nCheck tires."
    assert clean_text("cool-\ning system") == "cooling system"
    assert clean_text("2-\n3 bar") == "2-\n3 bar"


def test_chunks_have_token_bounds_overlap_and_page_provenance(pdf_path, tokenizer):
    pages, _ = extract_pdf(pdf_path)
    chunks, diagnostics = chunk_pages(pages, tokenizer, chunk_tokens=24, overlap_tokens=4)
    assert len(chunks) > 4
    assert diagnostics["empty_chunks"] == 0
    assert all(chunk.page_content.strip() for chunk in chunks)
    assert all(len(tokenizer.encode("passage: " + c.page_content)) <= 24 for c in chunks)
    assert {c.metadata["pdf_page"] for c in chunks} == {1, 2, 4}
    for chunk in chunks:
        page = pages[chunk.metadata["pdf_page"] - 1]
        assert (
            chunk.page_content
            == page.page_content[chunk.metadata["char_start"] : chunk.metadata["char_end"]]
        )
    first, second = chunks[:2]
    assert first.metadata["char_end"] > second.metadata["char_start"]


def test_suspicious_extraction_is_reported(tokenizer):
    from langchain_core.documents import Document

    pages = [Document(page_content="broken \ufffd glyph", metadata={"pdf_page": 1})]
    _, diagnostics = chunk_pages(pages, tokenizer)
    assert diagnostics["suspicious_chunks"][0]["pdf_page"] == 1
