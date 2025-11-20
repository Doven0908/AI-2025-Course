from __future__ import annotations

import math
import re
from typing import Iterable, List

from .types import DocumentChunk, RawDocument

SENTENCE_SEP_PATTERN = re.compile(r"(?<=[。！？?!])")


def _split_sentences(text: str) -> List[str]:
    """基于中文标点对文本进行粗粒度分句。"""
    sentences: List[str] = []
    buffer = ""
    for part in SENTENCE_SEP_PATTERN.split(text):
        if not part.strip():
            continue
        buffer += part
        if SENTENCE_SEP_PATTERN.search(part):
            sentences.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        sentences.append(buffer.strip())
    return sentences or [text]


def chunk_document(
    document: RawDocument,
    *,
    chunk_size: int,
    overlap: int,
) -> List[DocumentChunk]:
    """对单个文档执行滑动窗口分块，并保留重叠区域。"""

    sentences = _split_sentences(document.text)
    if not sentences:
        return []

    chunks: List[DocumentChunk] = []
    text_cursor = 0
    chunk_index = 0

    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = "".join(current_chunk).strip()
            start = max(0, text_cursor - current_length)
            end = start + len(chunk_text)
            chunk_id = f"{document.doc_id}_{chunk_index:05d}"
            chunks.append(
                DocumentChunk(
                    doc_id=document.doc_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    start_index=start,
                    end_index=end,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                    },
                )
            )

            chunk_index += 1
            overlap_chars = 0
            while current_chunk and current_length > overlap:
                removed = current_chunk.pop(0)
                current_length -= len(removed)
            current_length = sum(len(s) for s in current_chunk)

        current_chunk.append(sentence)
        current_length += sentence_length
        text_cursor += sentence_length

    if current_chunk:
        chunk_text = "".join(current_chunk).strip()
        start = max(0, text_cursor - current_length)
        end = start + len(chunk_text)
        chunk_id = f"{document.doc_id}_{chunk_index:05d}"
        chunks.append(
            DocumentChunk(
                doc_id=document.doc_id,
                chunk_id=chunk_id,
                text=chunk_text,
                start_index=start,
                end_index=end,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_index,
                },
            )
        )

    return chunks


def chunk_corpus(
    documents: Iterable[RawDocument],
    *,
    chunk_size: int,
    overlap: int,
) -> List[DocumentChunk]:
    """遍历整个语料库，返回全部分块结果。"""

    all_chunks: List[DocumentChunk] = []
    for document in documents:
        document_chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(document_chunks)
    return all_chunks
