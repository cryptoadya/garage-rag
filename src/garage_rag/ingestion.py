"""Page provenance, conservative text cleanup and tokenizer-bounded chunks."""

import hashlib
import re
import unicodedata
from collections import Counter
from io import BytesIO
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


class PDFValidationError(ValueError):
    """Invalid source PDF, with a fixed local diagnostic safe for CLI display."""


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))
    text = re.sub(r"(?<=[a-z])-\n(?=[a-z])", "", text)
    text = "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_pdf(path: Path, metadata: dict | None = None) -> tuple[list[Document], dict]:
    path = Path(path)
    raw = path.read_bytes()
    reader = PdfReader(BytesIO(raw), strict=False)
    if reader.is_encrypted and not reader.decrypt(""):
        raise PDFValidationError("Password-protected PDF is not supported")
    supplied = dict(metadata or {})
    # Invariant fields always come from the file, never user metadata.
    base = {
        "title": str((reader.metadata or {}).get("/Title") or path.stem),
        "language": "unknown",
        "vehicle_models": [],
        "document_type": "unknown",
        **supplied,
        "document": path.name,
        "document_id": hashlib.sha256(path.name.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
    }
    labels = reader.page_labels
    pages = [
        Document(
            page_content=clean_text(page.extract_text(extraction_mode="plain") or ""),
            metadata={**base, "pdf_page": i + 1, "page_label": labels[i]},
        )
        for i, page in enumerate(reader.pages)
    ]
    if not pages:
        raise PDFValidationError("PDF contains no pages")
    # Only repeated long margin lines: preserve short headings and all numeric lines.
    margins = Counter()
    for page in pages:
        lines = page.page_content.splitlines()
        margins.update(set(lines[:2] + lines[-2:]))
    repeated = {
        line
        for line, count in margins.items()
        if len(line) >= 35 and count >= max(3, len(pages) * 0.6)
    }
    for page in pages:
        lines = page.page_content.splitlines()
        page.page_content = "\n".join(
            line
            for i, line in enumerate(lines)
            if not (line in repeated and (i < 2 or i >= len(lines) - 2))
        ).strip()
    diagnostics = {
        "pages": len(pages),
        "pages_without_text": [p.metadata["pdf_page"] for p in pages if not p.page_content],
        "short_text_pages": [p.metadata["pdf_page"] for p in pages if 0 < len(p.page_content) < 80],
        "removed_repeated_margin_lines": len(repeated),
    }
    return pages, diagnostics


def chunk_pages(pages, tokenizer, chunk_tokens=384, overlap_tokens=48):
    if chunk_tokens < 16 or not 0 <= overlap_tokens < chunk_tokens // 2:
        raise ValueError("Invalid chunk token/overlap settings")
    prefix_cost = len(tokenizer.encode("passage: ", add_special_tokens=True))
    body_budget = chunk_tokens - prefix_cost - 2  # boundary tokenization safety margin
    if body_budget <= overlap_tokens:
        raise ValueError("Token budget too small for prefix and overlap")
    chunks = []
    suspicious = []
    for page in pages:
        text = page.page_content
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        start = 0
        ordinal = 0
        while start < len(offsets):
            end = min(start + body_budget, len(offsets))
            # Prefer a complete paragraph/sentence in the final quarter of the window.
            if end < len(offsets):
                for candidate in range(end, start + body_budget * 3 // 4, -1):
                    boundary = offsets[candidate - 1][1]
                    following = text[boundary : offsets[candidate][0]]
                    if "\n" in following or text[boundary - 1 : boundary] in ".!?":
                        end = candidate
                        break
            left = offsets[start][0]
            right = offsets[end - 1][1]
            # Tokenizer offsets can overlap for Unicode characters. Re-count exact input.
            while len(tokenizer.encode("passage: " + text[left:right])) > chunk_tokens:
                end -= 1
                if end <= start:
                    raise ValueError("Cannot fit a character in the chunk token budget")
                right = offsets[end - 1][1]
            content = text[left:right]
            if content.strip():
                item = Document(
                    page_content=content,
                    metadata={
                        **page.metadata,
                        "chunk_index": ordinal,
                        "char_start": left,
                        "char_end": right,
                        "tokens": len(tokenizer.encode("passage: " + content)),
                    },
                )
                chunks.append(item)
                reasons = []
                if "\ufffd" in content:
                    reasons.append("replacement_character")
                if sum(c.isalpha() for c in content) / len(content) < 0.3:
                    reasons.append("low_letter_ratio")
                if re.search(r"\S{100,}", content):
                    reasons.append("long_unbroken_text")
                if reasons:
                    suspicious.append(
                        {
                            "pdf_page": page.metadata["pdf_page"],
                            "chunk_index": ordinal,
                            "reasons": reasons,
                        }
                    )
                ordinal += 1
            if end == len(offsets):
                break
            start = max(start + 1, end - overlap_tokens)
    return chunks, {
        "empty_chunks": sum(not c.page_content.strip() for c in chunks),
        "suspicious_chunks": suspicious,
    }
