from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_openai import ChatOpenAI
from langchain_openrouter import ChatOpenRouter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from sentence_transformers import CrossEncoder

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler

from datetime import datetime
import os
import json
import numpy as np

from dotenv import load_dotenv
load_dotenv()


# Config 

QDRANT_URL      = "http://localhost:6333"
DOCS_COLLECTION = "epsilon_docs_githubs"

DOCS_K                = 10
DOCS_TOP_N_PER_QUERY  = 5



DOCS_SIBLING_MAX_PER_GROUP = 10
DOCS_FINAL_TOP_N = 8

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

RUN_SIMILARITY_DIAGNOSTIC = True

#  LLM selection 

llm_model = "qwen/qwen3.5-flash-02-23"

llm_provider = "openrouter"

# Token / cost tracking

PRICING_PER_1M_USD = {
    "gpt-5":              {"input": 1.25, "output": 10.00},
    "gpt-5-mini":         {"input": 0.25, "output": 2.00},
    "openai/gpt-5":       {"input": 1.25, "output": 10.00},
    "openai/gpt-5-mini":  {"input": 0.25, "output": 2.00},
}


# Embeddings 

def get_doc_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name   = "BAAI/bge-m3",
        model_kwargs = {"device": "cpu"},
        encode_kwargs= {
            "normalize_embeddings": True,
            "prompt": "Represent this sentence for searching relevant passages: ",
        },
    )


# get_code_embeddings() removed — docs-only variant


# Vector stores 

def get_docs_vectorstore() -> QdrantVectorStore:
    return QdrantVectorStore(
        client           = QdrantClient(url=QDRANT_URL),
        collection_name  = DOCS_COLLECTION,
        embedding        = get_doc_embeddings(),
        sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25"),
        retrieval_mode   = RetrievalMode.HYBRID,
    )



# Reranker 

def load_reranker() -> CrossEncoder:
    print(f"Loading reranker ({RERANKER_MODEL}) …")
    return CrossEncoder(RERANKER_MODEL)


def rerank_per_query(
    queries: list[str],
    results: list[tuple[Document, float]],
    reranker: CrossEncoder,
    top_n_per_query: int = 3,
) -> list[tuple[Document, float]]:
    if not results:
        return results

    parents    = [(doc, score) for doc, score in results if score == -1.0]
    candidates = [(doc, score) for doc, score in results if score != -1.0]

    if not candidates:
        return results

    seen_ids = set()
    merged   = []

    for query in queries:
        pairs  = [(query, doc.page_content[:512]) for doc, _ in candidates]
        scores = reranker.predict(pairs)

        top = sorted(
            zip([doc for doc, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n_per_query]

        for doc, score in top:
            uid = doc.metadata.get("chunk_id") or doc.page_content[:100]
            if uid not in seen_ids:
                seen_ids.add(uid)
                merged.append((doc, float(score)))

    merged.sort(key=lambda x: x[1], reverse=True)

    final = []
    child_class_names = {doc.metadata.get("class_name") for doc, _ in merged}
    for doc, score in parents:
        if doc.metadata.get("class_name") in child_class_names:
            final.append((doc, -1.0))

    final.extend(merged)
    return final


# Prompts

INTENT_PROMPT = ChatPromptTemplate.from_template(
    """Extract the core Eclipse Epsilon technical question from the user's message.
Ignore any sample data, YAML content, metamodel definitions, or code snippets the user pasted.
Focus only on what they want to know or achieve in Eclipse Epsilon terms.

Return the user's question verbatim as the first line, with only sample data and code snippets removed.
Then generate up to 2 additional search queries to retrieve relevant documentation and code examples.
At least one of the additional queries should focus on the Epsilon language syntax needed to accomplish the task.
Each additional query must relate directly to the same task described in the first line — do not introduce unrelated Epsilon languages or tools.
Each query must use only terms that would naturally appear in Eclipse Epsilon documentation — do not invent API names or class names.
Return one query per line, nothing else.

Message:
{message}

Queries:"""
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant answering questions about Eclipse Epsilon. Use only the documentation below — do not use outside knowledge, and say so if it doesn't cover something.

Documentation:
{context_block}

Before writing your answer, silently check each part of the question against the documentation — including whether any combination of separately-documented pieces is actually shown together, not just individually present somewhere. Don't show this checking process or split your answer into labeled parts. Write a smooth, natural answer, and weave in a brief, honest note wherever something isn't directly confirmed, rather than presenting it as verified.

Be concise, and let the length of your answer reflect how much the documentation actually supports — don't stretch a thin piece of evidence into a long, confident-sounding answer.

Question:
{question}

Answer:"""
)

# Token / cost tracking 

class TokenCostTracker(BaseCallbackHandler):

    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model    = model
        self.calls: list[dict] = []  # one record per LLM call this session
        self._dumped_once = False   # print raw usage structure once if tokens come back empty

    def on_llm_end(self, response, **kwargs):
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {}) or {}

        prompt_tokens     = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens       = token_usage.get("total_tokens", 0)
        raw_cost           = token_usage.get("cost", token_usage.get("total_cost"))

        if not prompt_tokens and not completion_tokens:
            try:
                message = response.generations[0][0].message
                usage_meta = getattr(message, "usage_metadata", None) or {}
                if usage_meta:
                    prompt_tokens     = usage_meta.get("input_tokens", 0)
                    completion_tokens = usage_meta.get("output_tokens", 0)
                    total_tokens       = usage_meta.get("total_tokens", prompt_tokens + completion_tokens)
                if raw_cost is None:
                    resp_meta = getattr(message, "response_metadata", None) or {}
                    raw_cost = resp_meta.get("cost")
                    token_usage_meta = resp_meta.get("token_usage", {}) or {}
                    raw_cost = raw_cost if raw_cost is not None else token_usage_meta.get("cost")
            except (AttributeError, IndexError):
                pass

        if not prompt_tokens and not completion_tokens and not self._dumped_once:
            self._dumped_once = True
            print("\n" + "=" * 60)
            print("DEBUG — full response.dump() (usage/cost path unknown, dumping everything):")
            try:
                print(json.dumps(response.model_dump(), indent=2, default=str))
            except AttributeError:
                try:
                    print(json.dumps(response.dict(), indent=2, default=str))
                except AttributeError:
                    print(repr(response))
            print("=" * 60 + "\n")

        cost_usd    = None
        cost_source = None

        if self.provider == "openrouter" and raw_cost is not None:
            cost_usd    = float(raw_cost)
            cost_source = "openrouter_usage_accounting"

        if cost_usd is None:
            pricing = PRICING_PER_1M_USD.get(self.model)
            if pricing:
                cost_usd = (
                    prompt_tokens / 1_000_000 * pricing["input"]
                    + completion_tokens / 1_000_000 * pricing["output"]
                )
                cost_source = "manual_pricing_table"
            else:
                cost_usd    = 0.0
                cost_source = "unknown_model_no_pricing"

        self.calls.append({
            "prompt_tokens":     prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":      total_tokens,
            "cost_usd":          cost_usd,
            "cost_source":       cost_source,
        })

    def turn_summary(self, since_call_index: int) -> dict:
        """Aggregate calls made from since_call_index to the end (i.e. this turn)."""
        turn_calls = self.calls[since_call_index:]
        return {
            "calls":             len(turn_calls),
            "prompt_tokens":     sum(c["prompt_tokens"] for c in turn_calls),
            "completion_tokens": sum(c["completion_tokens"] for c in turn_calls),
            "total_tokens":      sum(c["total_tokens"] for c in turn_calls),
            "cost_usd":          sum(c["cost_usd"] for c in turn_calls),
        }

    def session_summary(self) -> dict:
        return self.turn_summary(0)


# Input

def read_message() -> str:
    print("Ask (type END on a new line to submit):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines).strip()


# Pipeline steps

def extract_queries(message: str, llm, tracker: TokenCostTracker) -> list[str]:
    chain = INTENT_PROMPT | llm | StrOutputParser()
    result = chain.invoke(
        {"message": message},
        config={"callbacks": [tracker]},
    ).strip()
    queries = [q.strip() for q in result.split("\n") if q.strip()]
    return queries


def retrieve_with_scores(
    query: str,
    docs_vectorstore: QdrantVectorStore,
) -> list[tuple[Document, float]]:
    doc_results = docs_vectorstore.similarity_search_with_score(query, k=DOCS_K)
    return doc_results


def retrieve_all_queries(
    queries: list[str],
    docs_vectorstore: QdrantVectorStore,
) -> list[tuple[Document, float]]:
    seen_doc_ids = set()
    doc_results  = []

    for q in queries:
        d_res = retrieve_with_scores(q, docs_vectorstore)

        for doc, score in d_res:
            uid = doc.metadata.get("chunk_id") or doc.page_content[:100]
            if uid not in seen_doc_ids:
                seen_doc_ids.add(uid)
                doc_results.append((doc, score))

    return doc_results


def expand_with_doc_siblings(
    doc_results: list[tuple[Document, float]],
    qdrant_client: QdrantClient,
    max_per_group: int = DOCS_SIBLING_MAX_PER_GROUP,
) -> list[tuple[Document, float]]:
    if not doc_results:
        return doc_results

    seen_keys = {
        (doc.metadata.get("source"), doc.page_content[:100])
        for doc, _ in doc_results
    }

    expanded = list(doc_results)
    seen_groups = set()  # (source, parent_heading) — avoid re-querying the same group twice

    for doc, _score in doc_results:
        source         = doc.metadata.get("source", "")
        parent_heading = doc.metadata.get("parent_heading", "")

        # nothing to group by — skip (e.g. top-level intro chunks with no parent_heading)
        if not source or not parent_heading:
            continue

        group_key = (source, parent_heading)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)

        results, _ = qdrant_client.scroll(
            collection_name = DOCS_COLLECTION,
            scroll_filter   = Filter(must=[
                FieldCondition(key="metadata.source",         match=MatchValue(value=source)),
                FieldCondition(key="metadata.parent_heading",  match=MatchValue(value=parent_heading)),
            ]),
            limit        = max_per_group,
            with_payload = True,
            with_vectors = False,
        )

        for r in results:
            payload = r.payload
            key = (payload["metadata"].get("source"), payload["page_content"][:100])
            if key not in seen_keys:
                seen_keys.add(key)
                expanded.append((
                    Document(page_content=payload["page_content"], metadata=payload["metadata"]),
                    0.0,  # placeholder — treated as a normal candidate in the next rerank pass
                ))

    return expanded


def dedup_by_heading(
    results: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    seen_headings = set()
    deduped = []

    for doc, score in results:
        if score == -1.0:
            deduped.append((doc, score))
            continue

        heading = doc.metadata.get("heading", "")
        if not heading:
            deduped.append((doc, score))
            continue

        key = (doc.metadata.get("source", ""), heading)
        if key in seen_headings:
            continue
        seen_headings.add(key)
        deduped.append((doc, score))

    return deduped


def build_context_block(
    message: str,
    doc_results: list[tuple[Document, float]],
) -> str:
    parts = []

    if doc_results:
        docs_text = "\n\n".join(
            f"[Source: {d.metadata.get('source', 'N/A')}]\n{d.page_content}"
            for d, _ in doc_results
        )
        parts.append(f"--- DOCUMENTATION ---\n{docs_text}")

    # code context block removed — docs-only variant

    parts.append(f"--- USER MESSAGE ---\n{message}")

    return "\n\n".join(parts)


# Similarity diagnostic

def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def log_query_similarity_diagnostic(
    log,
    queries: list[str],
    doc_results: list[tuple[Document, float]],
    embeddings_model: HuggingFaceEmbeddings,
):
    if not doc_results:
        return

    log(f"\n── SIMILARITY DIAGNOSTIC (dense cosine, pre-rerank, full ranking) ──")

    # dedupe candidate chunks by heading so we don't embed the same chunk repeatedly
    candidates = {}
    for doc, _ in doc_results:
        key = (doc.metadata.get("source", ""), doc.metadata.get("heading", "") or doc.page_content[:100])
        if key not in candidates:
            candidates[key] = doc

    candidate_list = list(candidates.values())
    candidate_vecs = embeddings_model.embed_documents(
        [d.page_content[:512] for d in candidate_list]
    )

    for query in queries:
        q_vec = embeddings_model.embed_query(query)
        sims = [
            (doc, cosine_sim(q_vec, vec))
            for doc, vec in zip(candidate_list, candidate_vecs)
        ]
        sims.sort(key=lambda x: x[1], reverse=True)

        log(f"\n  Query: {query!r}")
        for i, (doc, sim) in enumerate(sims, 1):
            heading = doc.metadata.get("heading", "N/A")
            source  = doc.metadata.get("source", "N/A")
            log(f"    {i}. sim={sim:.4f}  {source} | heading={heading!r}")


# Logging 

def make_logger(log_path: str):
    def log(text: str):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")
            f.flush()
    return log


def log_retrieved(
    log,
    doc_results: list[tuple[Document, float]],
):
    if doc_results:
        log(f"\nDocs ({len(doc_results)}):")
        for i, (doc, score) in enumerate(doc_results, 1):
            score_str = f"{score:.3f}" if score >= 0 else "parent"
            log(f"  {i}. [{score_str}] {doc.metadata.get('source', 'N/A')} "
                f"(heading={doc.metadata.get('heading', '')!r}, "
                f"parent_heading={doc.metadata.get('parent_heading', '')!r})")
            log(f"     {doc.page_content[:200]}")


def log_turn_cost(log, turn: dict):
    log(f"\n── TOKENS / COST (this turn) ─────────────────────────────")
    log(f"  Calls:             {turn['calls']}")
    log(f"  Prompt tokens:     {turn['prompt_tokens']:,}")
    log(f"  Completion tokens: {turn['completion_tokens']:,}")
    log(f"  Total tokens:      {turn['total_tokens']:,}")
    log(f"  Estimated cost:    ${turn['cost_usd']:.5f}")


def log_session_summary(log, tracker: TokenCostTracker):
    summary = tracker.session_summary()
    log(f"\n{'=' * 60}")
    log(f"SESSION SUMMARY — {tracker.provider}/{tracker.model}")
    log(f"  Total LLM calls:         {summary['calls']}")
    log(f"  Total prompt tokens:     {summary['prompt_tokens']:,}")
    log(f"  Total completion tokens: {summary['completion_tokens']:,}")
    log(f"  Total tokens:            {summary['total_tokens']:,}")
    log(f"  Total estimated cost:    ${summary['cost_usd']:.5f}")
    log("=" * 60)


def print_turn_cost(turn: dict):
    print(f"  [tokens] prompt={turn['prompt_tokens']:,} "
          f"completion={turn['completion_tokens']:,} "
          f"total={turn['total_tokens']:,}  "
          f"cost=${turn['cost_usd']:.5f}")


def print_session_summary(tracker: TokenCostTracker):
    summary = tracker.session_summary()
    print(f"\n{'=' * 60}")
    print(f"SESSION SUMMARY — {tracker.provider}/{tracker.model}")
    print(f"  Total LLM calls:         {summary['calls']}")
    print(f"  Total prompt tokens:     {summary['prompt_tokens']:,}")
    print(f"  Total completion tokens: {summary['completion_tokens']:,}")
    print(f"  Total tokens:            {summary['total_tokens']:,}")
    print(f"  Total estimated cost:    ${summary['cost_usd']:.5f}")
    print("=" * 60)


def print_retrieved(
    doc_results: list[tuple[Document, float]],
):
    if doc_results:
        print(f"\nDocs ({len(doc_results)}):")
        for i, (doc, score) in enumerate(doc_results, 1):
            score_str = f"{score:.3f}" if score >= 0 else "parent"
            print(f"  {i}. [{score_str}] {doc.metadata.get('source', 'N/A')} "
                  f"(heading={doc.metadata.get('heading', '')!r})")
            print(f"     {doc.page_content[:500]}")


# Main

def main():
    print("Loading vector stores …")
    docs_vectorstore = get_docs_vectorstore()
    qdrant_client    = QdrantClient(url=QDRANT_URL)
    reranker         = load_reranker()
    doc_embeddings   = get_doc_embeddings()  # reused for similarity diagnostic

    if llm_provider == "openai":
        llm = ChatOpenAI(model=llm_model, temperature=0)
    elif llm_provider == "openrouter":
        llm = ChatOpenRouter(model=llm_model, temperature=0)
    else:
        raise ValueError(f"Unknown llm_provider: {llm_provider!r}")

    tracker = TokenCostTracker(provider=llm_provider, model=llm_model)

    os.makedirs("chat_logs", exist_ok=True)

    # one timestamped file per session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join("chat_logs", f"chat_{timestamp}.txt")
    log       = make_logger(log_path)

    print(f"RAG chat started. Model: {llm_provider}/{llm_model}. Log → {log_path}")
    print("Type 'exit' or 'quit' then END to end the session.\n")

    log(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Model: {llm_provider}/{llm_model} | temperature=0")
    log(f"DOCS_K={DOCS_K} DOCS_FINAL_TOP_N={DOCS_FINAL_TOP_N} "
        f"DOCS_TOP_N_PER_QUERY={DOCS_TOP_N_PER_QUERY}")
    log("=" * 60)

    try:
        while True:
            message = read_message()

            if not message:
                continue
            if message.lower() in {"exit", "quit"}:
                log("\n[Session ended]")
                break

            call_index_before_turn = len(tracker.calls)

            # ── 1. extract queries for retrieval ──
            queries = extract_queries(message, llm, tracker)
            print(f"  [queries] {queries}")

            # ── 2. retrieve docs using all queries, deduplicated ──
            doc_results = retrieve_all_queries(queries, docs_vectorstore)

            # DEBUG — pool ก่อน rerank
            print(f"\n[DEBUG] Doc pool before rerank ({len(doc_results)} docs):")
            for i, (doc, score) in enumerate(doc_results, 1):
                print(f"  {i}. [{score:.3f}] {doc.metadata.get('source', 'N/A')}")

            # ── 2b. similarity diagnostic on the raw pre-rerank doc pool ──
            log(f"\n{'=' * 60}")
            log(f"[{datetime.now().strftime('%H:%M:%S')}]")
            log(f"\n── QUESTION ──────────────────────────────────────────────")
            log(message)
            log(f"\n── QUERIES (used for retrieval) ──────────────────────────")
            for q in queries:
                log(f"  - {q}")

            if RUN_SIMILARITY_DIAGNOSTIC:
                log_query_similarity_diagnostic(
                    log, queries, doc_results, doc_embeddings,
                )

            # ── 3. rerank per query — pass 1 (top N per query, then merge + dedup) ──
            doc_results = rerank_per_query(queries, doc_results, reranker, top_n_per_query=DOCS_TOP_N_PER_QUERY)
            doc_results = dedup_by_heading(doc_results)

            # ── 3b. expand top-ranked docs with sibling chunks (same H2 section) ──
            if doc_results:
                doc_results = expand_with_doc_siblings(doc_results, qdrant_client)

                # ── 3c. rerank per query — pass 2: cross-encoder filters which
                #        siblings are actually relevant, instead of auto-including all ──
                doc_results = rerank_per_query(queries, doc_results, reranker, top_n_per_query=DOCS_TOP_N_PER_QUERY)
                doc_results = dedup_by_heading(doc_results)
                doc_results = doc_results[:DOCS_FINAL_TOP_N]

            # ── 4. build full prompt (full message goes to LLM, not queries) ──
            context_block    = build_context_block(message, doc_results)
            full_prompt      = ANSWER_PROMPT.format_messages(
                context_block = context_block,
                question      = message,
            )
            full_prompt_text = "\n".join(
                f"[{m.type.upper()}]\n{m.content}" for m in full_prompt
            )

            # ── 5. generate ──
            chain    = ANSWER_PROMPT | llm | StrOutputParser()
            response = chain.invoke(
                {
                    "context_block": context_block,
                    "question"     : message,
                },
                config={"callbacks": [tracker]},
            )

            turn = tracker.turn_summary(call_index_before_turn)

            # terminal output
            print(f"\nAnswer:\n{response}")
            print_retrieved(doc_results)
            print_turn_cost(turn)
            print("-" * 60)

            # log everything else
            log(f"\n── RETRIEVED (after dedup + sibling expansion + rerank) ──")
            log_retrieved(log, doc_results)

            log(f"\n── FULL PROMPT SENT TO LLM ───────────────────────────────")
            log(full_prompt_text)

            log(f"\n── ANSWER ────────────────────────────────────────────────")
            log(response)

            log_turn_cost(log, turn)
            log("-" * 60)

    finally:
        # Runs on normal 'exit'/'quit' AND on Ctrl+C / crashes, so the
        # session total is never lost even if the session ends abruptly.
        log_session_summary(log, tracker)
        print_session_summary(tracker)


if __name__ == "__main__":
    main()