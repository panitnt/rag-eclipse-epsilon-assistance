import re
import glob
import os
import math
from collections import defaultdict
import pandas as pd

# CONFIG — edit these

METHOD_NAME = "sparse_sibON" # just a label, used in printed output / CSV filenames
LOG_DIR = "chat_logs" # folder of .txt logs to evaluate

K_VALUES = [1, 3, 5, 8, 10]

# Both sections live in the SAME log file. Listing more than one here runs
# metrics for each and saves separate CSVs per section — no need to edit
# and re-run per layer:
#   "pool"       -> raw pre-rerank pool block: "DENSE POOL (before rerank,
#                   before sibling expansion)" / "SPARSE POOL (...)" /
#                   "HYBRID POOL (...)" — whichever is present. Clean
#                   retrieval-only signal (no rerank/expansion/dedup yet).
#   "retrieved"  -> "RETRIEVED (after dedup + sibling expansion + rerank)"
#                   block — the FINAL context actually sent to the LLM,
#                   after rerank pass 1 -> sibling expansion -> rerank pass 2.
#                   Useful to check whether sibling expansion masked
#                   differences that showed up at the "pool" layer.
#   "diagnostic" -> "SIMILARITY DIAGNOSTIC" block (dense cosine only,
#                   always present). Multi-query "Query:" sub-blocks are
#                   fused by max similarity per chunk.
SECTIONS_TO_RUN = ["pool", "retrieved"]

PATH_YAML = "epsilon_docs/articles_yaml-emc_index.md"
PATH_INDEX = "epsilon_docs/articles_index.md"
PATH_EMC = "epsilon_docs/emc.md"
PATH_EOL = "epsilon_docs/eol.md"


def chunk_id(path, heading):
    """Unique key for a (source file, heading) chunk.
    NOTE: adjust this if chunk_id is defined differently elsewhere in
    your codebase — it must match exactly or gold lookups will silently
    miss."""
    return f"{path}::{heading.strip()}"


GOLD = {
    # --- Primary (1.0) ---
    chunk_id(PATH_YAML, "How can I create a YAML document from scratch?"): 1.0,
    chunk_id(PATH_YAML, "How do I set the root node of a YAML document?"): 1.0,
    chunk_id(PATH_YAML, "How can I create a node?"): 1.0,
    # --- Secondary (0.5) ---
    # chunk_id(PATH_YAML, "Querying a YAML document"): 0.5,
    # chunk_id(PATH_YAML, "How can I access all node elements?"): 0.5,
    # chunk_id(PATH_YAML, "How can I delete nodes?"): 0.5,
    # chunk_id(PATH_INDEX, "Recipes"): 0.5,
    # chunk_id(PATH_EMC, "Other Drivers"): 0.5,
    # chunk_id(PATH_EOL, "Creating and Deleting Model Elements"): 0.5,
}

# LOG PARSING

# e.g. "  1. sim=0.8101  epsilon_docs/articles_yaml-emc_index.md | heading='...'"
DIAG_LINE_RE = re.compile(
    r"^\s*\d+\.\s+sim=([\d.]+)\s+(\S+)\s*\|\s*heading='([^']*)'"
)
# e.g. "  1. [0.982] epsilon_docs/articles_yaml-emc_index.md (heading='...', parent_heading='...')"
# Matches lines in both the POOL blocks and the RETRIEVED block — same format.
POOL_LINE_RE = re.compile(
    r"^\s*\d+\.\s+\[([\d.]+|parent)\]\s+(\S+)\s+\(heading='([^']*)'"
)


def _dedup_keep_max(pairs):
    """pairs: list of (chunk_id, score). Collapse duplicates by keeping the
    max score per chunk_id, then return chunk_ids sorted by that score,
    descending. Needed because multi-query retrieval can return the same
    chunk more than once (once per sub-query) before any dedup step runs."""
    best = {}
    for cid, score in pairs:
        if cid not in best or score > best[cid]:
            best[cid] = score
    return [cid for cid, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)]


def parse_log(filepath, section="pool"):
    """Return an ordered list of chunk_id strings (rank 1 first) parsed
    from one log file."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    if section == "pool":
        block_match = re.search(
            r"── (?:DENSE|SPARSE|HYBRID) POOL \(before rerank.*?\) ──(.*?)"
            r"(?:── SIMILARITY DIAGNOSTIC|── RETRIEVED)",
            text, re.S,
        )
        if not block_match:
            return []
        pairs = []
        for line in block_match.group(1).splitlines():
            m = POOL_LINE_RE.match(line)
            if m:
                score_str, path, heading = m.groups()
                score = -1.0 if score_str == "parent" else float(score_str)
                pairs.append((chunk_id(path, heading), score))
        return _dedup_keep_max(pairs)

    elif section == "diagnostic":
        block_match = re.search(
            r"── SIMILARITY DIAGNOSTIC.*?──(.*?)(?:── RETRIEVED|\Z)", text, re.S
        )
        if not block_match:
            return []
        pairs = []
        for line in block_match.group(1).splitlines():
            m = DIAG_LINE_RE.match(line)
            if m:
                sim_str, path, heading = m.groups()
                pairs.append((chunk_id(path, heading), float(sim_str)))
        return _dedup_keep_max(pairs)

    elif section == "retrieved":
        block_match = re.search(
            r"── RETRIEVED.*?──(.*?)── FULL PROMPT SENT TO LLM", text, re.S
        )
        if not block_match:
            return []
        ranked = []
        for line in block_match.group(1).splitlines():
            m = POOL_LINE_RE.match(line)
            if m:
                _, path, heading = m.groups()
                ranked.append(chunk_id(path, heading))
        return ranked

    else:
        raise ValueError(f"Unknown section: {section}")


# METRICS

def precision_at_k(ranked_ids, gold, k):
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for cid in top_k if gold.get(cid, 0) > 0)
    return relevant / len(top_k)


def recall_at_k(ranked_ids, gold, k):
    total_relevant = sum(1 for v in gold.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    top_k = ranked_ids[:k]
    relevant_found = sum(1 for cid in top_k if gold.get(cid, 0) > 0)
    return relevant_found / total_relevant


def average_precision(ranked_ids, gold):
    """Standard AP, binary relevance (gold score > 0 counts as relevant)."""
    total_relevant = sum(1 for v in gold.values() if v > 0)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precisions = []
    for i, cid in enumerate(ranked_ids, start=1):
        if gold.get(cid, 0) > 0:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    return sum(precisions) / total_relevant


def dcg_at_k(ranked_ids, gold, k):
    dcg = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        rel = gold.get(cid, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids, gold, k):
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum(
        rel / math.log2(i + 1) for i, rel in enumerate(ideal, start=1) if rel > 0
    )
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, gold, k) / idcg


# MAIN

def run_section(section):
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*.txt")))
    if not files:
        print(f"No .txt files found in {LOG_DIR}")
        return

    per_k_precision = defaultdict(list)
    per_k_recall = defaultdict(list)
    per_k_ndcg = defaultdict(list)
    ap_scores = []
    details = []

    for fp in files:
        ranked = parse_log(fp, section=section)
        if not ranked:
            print(f"  [!] no ranked chunks parsed from {fp} (section='{section}') — skipping")
            continue

        row = {"file": os.path.basename(fp), "n_chunks": len(ranked)}
        for k in K_VALUES:
            p = precision_at_k(ranked, GOLD, k)
            r = recall_at_k(ranked, GOLD, k)
            n = ndcg_at_k(ranked, GOLD, k)
            per_k_precision[k].append(p)
            per_k_recall[k].append(r)
            per_k_ndcg[k].append(n)
            row[f"P@{k}"] = round(p, 4)
            row[f"R@{k}"] = round(r, 4)
            row[f"nDCG@{k}"] = round(n, 4)

        ap = average_precision(ranked, GOLD)
        ap_scores.append(ap)
        row["AP"] = round(ap, 4)
        details.append(row)

    if not ap_scores:
        print(f"Nothing parsed for section='{section}' — check LOG_DIR / log format.")
        return

    n = len(ap_scores)
    summary = {"n_queries": n, "MAP": sum(ap_scores) / n}
    for k in K_VALUES:
        summary[f"P@{k}"] = sum(per_k_precision[k]) / len(per_k_precision[k])
        summary[f"R@{k}"] = sum(per_k_recall[k]) / len(per_k_recall[k])
        summary[f"nDCG@{k}"] = sum(per_k_ndcg[k]) / len(per_k_ndcg[k])

    summary_df = pd.DataFrame([summary], index=[METHOD_NAME]).round(4)
    details_df = pd.DataFrame(details)

    print("\n" + "=" * 60)
    print(f"RETRIEVAL EVAL — {METHOD_NAME}  (section='{section}', {LOG_DIR})")
    print("=" * 60)
    print(summary_df.to_string())

    print(f"\nPer-file breakdown ({len(details)} files):")
    print(details_df.to_string(index=False))

    summary_path = f"30_only_1/30_retrieve_result_other_question/retrieval_eval_{METHOD_NAME}_{section}_summary.csv"
    details_path = f"30_only_1/30_retrieve_result_other_question/retrieval_eval_{METHOD_NAME}_{section}_details.csv"
    summary_df.to_csv(summary_path)
    details_df.to_csv(details_path, index=False)
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {details_path}")


def main():
    for section in SECTIONS_TO_RUN:
        run_section(section)


if __name__ == "__main__":
    main()