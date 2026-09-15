"""
Epsilon Docs Embedder
Reads .md files from epsilon_docs/, chunks with code-awareness,
and upserts into Qdrant collection: epsilon_docs

Chunking strategy:
  - prose      -> split by RecursiveCharacter (max 1500 chars)
  - code blocks -> never split, stored as one chunk with:
        1. heading prepended
        2. LLM-generated intent summary prepended
        3. keyword list prepended
    so embedding captures both intent and identifiers

Metadata per chunk:
  source         — file path
  type           — "prose" or "code"
  heading        — nearest H1/H2/H3 above this chunk
  parent_heading — nearest heading of higher level
"""

import re
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams

from dotenv import load_dotenv
load_dotenv()


### Config

DOCS_DIR            = "epsilon_docs"
QDRANT_URL          = "http://localhost:6333"
COLLECTION_NAME     = "epsilon_docs_githubs"
PROSE_CHUNK_SIZE    = 1500
PROSE_CHUNK_OVERLAP = 200
BATCH_SIZE          = 100


### Heading extractor

_HEADING_RE = re.compile(r'^(#{1,3}) (.+)', re.MULTILINE)


def _heading_level(raw: str) -> int:
    """Return 1/2/3 from a raw heading string like '## Foo'."""
    m = re.match(r'^(#{1,3})', raw)
    return len(m.group(1)) if m else 0


def _strip_hashes(raw: str) -> str:
    """'## Loading Models' -> 'Loading Models'"""
    return re.sub(r'^#{1,3}\s*', '', raw).strip()


def _headings_before(parts: list[str], current_idx: int) -> dict[str, str]:
    """
    Walk backwards through prose parts before current_idx and collect
    the most recent heading at each level (H1, H2, H3).
    Returns {"heading": <nearest>, "parent_heading": <nearest of higher level>}.

    FIX: use max(level_map) to get the deepest (nearest) heading,
         not min(level_map) which was incorrectly returning H1 every time.
    """
    level_map: dict[int, str] = {}

    for part in reversed(parts[:current_idx]):
        if part.startswith("```"):
            continue
        for m in reversed(list(_HEADING_RE.finditer(part))):
            raw   = m.group(0)
            level = _heading_level(raw)
            if level not in level_map:
                level_map[level] = raw
        if len(level_map) >= 3:
            break

    if not level_map:
        return {"heading": "", "parent_heading": ""}

    # max -> deepest level = nearest heading (e.g. H3 over H1)
    nearest_level = max(level_map)
    nearest       = _strip_hashes(level_map[nearest_level])

    parent_levels = [l for l in level_map if l < nearest_level]
    parent        = _strip_hashes(level_map[max(parent_levels)]) if parent_levels else ""

    return {"heading": nearest, "parent_heading": parent}


### Keyword extractor 

_IDENTIFIER_RE = re.compile(r'\b([A-Z][a-zA-Z]+Node|[a-z]+Node|appendNode|addRow[s]?|'
                             r'setRootAs(?:List|Map)|getRoot|ScalarNode|MappingNode|'
                             r'ListNode|\.value|\.name|\.type|new\s+\w+)\b')


def _extract_keywords(code: str) -> str:
    seen = []
    for m in _IDENTIFIER_RE.finditer(code):
        token = m.group(1).strip()
        if token not in seen:
            seen.append(token)
    return ", ".join(seen[:20])


### LLM intent summarizer

_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are indexing Eclipse Epsilon documentation for a RAG system.
Given the heading and code block below, write ONE sentence (max 20 words)
describing what this code example demonstrates.
Start with "Example:" — e.g. "Example: create a YAML document from scratch using MappingNode, ListNode, and appendNode."
Return only that sentence, nothing else.

Heading: {heading}
Code:
{code}

Summary:"""
)

_summarizer = None


def get_summary(heading: str, code: str) -> str:
    global _summarizer
    if _summarizer is None:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        _summarizer = _SUMMARY_PROMPT | llm | StrOutputParser()
    try:
        return _summarizer.invoke({"heading": heading, "code": code[:800]}).strip()
    except Exception:
        return ""


### Code-aware chunker

_CODE_BLOCK_RE = re.compile(r'(```[\w]*\n.*?```)', re.DOTALL)


def chunk_markdown(text: str, source: str) -> list[Document]:
    """
    Splits markdown into chunks:
    - code blocks -> 1 chunk each, prepended with heading + LLM summary + keywords
    - prose       -> split by RecursiveCharacterTextSplitter

    Every chunk receives metadata:
        source, type, heading, parent_heading
    """
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size    = PROSE_CHUNK_SIZE,
        chunk_overlap = PROSE_CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = []
    parts = _CODE_BLOCK_RE.split(text)
    prose_so_far = ""
    current_headings: dict[str, str] = {"heading": "", "parent_heading": ""}

    for idx, part in enumerate(parts):
        if part.startswith("```"):
            # flush pending prose
            if prose_so_far.strip():
                for chunk_text in char_splitter.split_text(prose_so_far):
                    if chunk_text.strip():
                        chunks.append(Document(
                            page_content = chunk_text.strip(),
                            metadata     = {
                                "source":         source,
                                "type":           "prose",
                                "heading":        current_headings["heading"],
                                "parent_heading": current_headings["parent_heading"],
                            },
                        ))
                prose_so_far = ""

            # resolve headings for this code chunk 
            headings = _headings_before(parts, idx)
            heading  = headings["heading"]
            summary  = get_summary(heading, part)
            keywords = _extract_keywords(part)

            prefix_parts = []
            if heading:
                prefix_parts.append(f"## {heading}")
            if summary:
                prefix_parts.append(summary)
            if keywords:
                prefix_parts.append(f"Keywords: {keywords}")

            prefix  = "\n".join(prefix_parts)
            content = f"{prefix}\n\n{part}".strip() if prefix else part.strip()

            chunks.append(Document(
                page_content = content,
                metadata     = {
                    "source":         source,
                    "type":           "code",
                    "heading":        heading,
                    "parent_heading": headings["parent_heading"],
                },
            ))

        else:
            # update running heading context from this prose segment
            if list(_HEADING_RE.finditer(part)):
                current_headings = _headings_before(parts, idx + 1)
            prose_so_far += part

    # flush remaining prose
    if prose_so_far.strip():
        for chunk_text in char_splitter.split_text(prose_so_far):
            if chunk_text.strip():
                chunks.append(Document(
                    page_content = chunk_text.strip(),
                    metadata     = {
                        "source":         source,
                        "type":           "prose",
                        "heading":        current_headings["heading"],
                        "parent_heading": current_headings["parent_heading"],
                    },
                ))

    return chunks


### Load

def load_and_chunk(docs_dir: str) -> list[Document]:
    md_files = sorted(Path(docs_dir).rglob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {docs_dir}")

    all_chunks: list[Document] = []
    failed: list[str] = []

    for path in md_files:
        try:
            text   = path.read_text(encoding="utf-8", errors="replace")
            source = str(path)
            chunks = chunk_markdown(text, source)
            all_chunks.extend(chunks)
            prose_n = sum(1 for c in chunks if c.metadata["type"] == "prose")
            code_n  = sum(1 for c in chunks if c.metadata["type"] == "code")
            print(f"  {path.name:<55} prose={prose_n}  code={code_n}")
        except Exception as e:
            print(f"  ✗ {path.name}: {e}")
            failed.append(str(path))

    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")

    return all_chunks


### Qdrant helpers

def ensure_collection(client: QdrantClient, name: str):
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        client.delete_collection(name)
        print(f"  Deleted old collection: {name}")

    client.create_collection(
        collection_name       = name,
        vectors_config        = VectorParams(size=1024, distance=Distance.COSINE),
        sparse_vectors_config = {
            "langchain-sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"  Created collection: {name}")


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name   = "BAAI/bge-m3",
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {"normalize_embeddings": True},
    )


def upsert(chunks: list[Document]):
    embeddings        = get_embeddings()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    client            = QdrantClient(url=QDRANT_URL)

    ensure_collection(client, COLLECTION_NAME)

    print(f"\n  Upserting {len(chunks)} chunks in batches of {BATCH_SIZE} …")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]
        QdrantVectorStore.from_documents(
            documents        = batch,
            embedding        = embeddings,
            sparse_embedding = sparse_embeddings,
            url              = QDRANT_URL,
            collection_name  = COLLECTION_NAME,
            retrieval_mode   = RetrievalMode.HYBRID,
        )
        print(f"    [{i + len(batch)}/{len(chunks)}]")


### Preview helper

def preview_chunks(chunks: list[Document], n: int = 3):
    print("\n── Preview (code chunks) ───────────────────────────────")
    code_chunks = [c for c in chunks if c.metadata["type"] == "code"]
    for c in code_chunks[:n]:
        print(f"  source         : {c.metadata['source']}")
        print(f"  heading        : {c.metadata['heading']}")
        print(f"  parent_heading : {c.metadata['parent_heading']}")
        print(f"  content        : {c.page_content[:400]}")
        print()


### Main

def main():
    print(f"Loading & chunking markdown files from '{DOCS_DIR}' …\n")
    chunks = load_and_chunk(DOCS_DIR)

    prose_total = sum(1 for c in chunks if c.metadata["type"] == "prose")
    code_total  = sum(1 for c in chunks if c.metadata["type"] == "code")
    print(f"\nTotal chunks: {len(chunks)}  (prose={prose_total}, code={code_total})")

    preview_chunks(chunks)

    print("Building Qdrant vector store …")
    upsert(chunks)

    print(f"\nTotal: {len(chunks)} chunks in '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()