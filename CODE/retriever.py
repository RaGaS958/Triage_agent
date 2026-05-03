"""
retriever.py — BM25-based corpus retriever with company-scoped search.

Two-stage retrieval:
  1. Company-scoped BM25 search (filters docs by product domain first)
  2. Global BM25 fallback if scoped search yields low confidence
"""

import os
import re
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi


@dataclass
class CorpusDoc:
    doc_id: str
    path: str
    company: str          # "hackerrank" | "claude" | "visa"
    section: str          # top-level section from path
    title: str
    content: str
    tokens: List[str] = field(default_factory=list)


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _extract_title(content: str, path: str) -> str:
    """Extract title from frontmatter or first H1."""
    # frontmatter title
    m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    # first H1
    m = re.search(r'^#\s+(.+)', content, re.MULTILINE)
    if m:
        return m.group(1).strip()
    # fallback: filename
    return Path(path).stem.replace("-", " ")


def _company_from_path(path: str) -> str:
    """Detect company from the path segment AFTER the 'data/' directory."""
    # Find the data/ segment and look at what follows it
    parts = Path(path).parts
    try:
        # Find 'data' in parts (case-insensitive)
        idx = next(i for i, p in enumerate(parts) if p.lower() == "data")
        if len(parts) > idx + 1:
            company_seg = parts[idx + 1].lower()
            if "hackerrank" in company_seg:
                return "hackerrank"
            if "claude" in company_seg:
                return "claude"
            if "visa" in company_seg:
                return "visa"
    except StopIteration:
        pass
    return "unknown"


def _section_from_path(path: str) -> str:
    """Return the top-level section name (e.g. 'screen', 'account-management')."""
    parts = Path(path).parts
    # Find 'data' in parts and take 2 levels after
    try:
        idx = [p.lower() for p in parts].index("data")
        # parts[idx+1] = company, parts[idx+2] = top section
        if len(parts) > idx + 2:
            return parts[idx + 2]
    except ValueError:
        pass
    return "general"


class CorpusRetriever:
    """
    Loads the full support corpus from data/ and provides BM25 retrieval.
    Supports:
      - Company-scoped search
      - Global search
      - Confidence scoring
    """

    MIN_SCORE = 0.5   # below this → low confidence

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.docs: List[CorpusDoc] = []
        self._company_index: Dict[str, Tuple[BM25Okapi, List[CorpusDoc]]] = {}
        self._global_bm25: Optional[BM25Okapi] = None

        self._load_corpus()
        self._build_indices()

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def _load_corpus(self):
        data_path = Path(self.data_dir)
        md_files = list(data_path.rglob("*.md"))
        for md_file in md_files:
            try:
                raw = md_file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            path_str = str(md_file)
            company = _company_from_path(path_str)
            section = _section_from_path(path_str)
            title = _extract_title(raw, path_str)

            # Strip YAML frontmatter
            content = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL)
            # Strip markdown image syntax (noise)
            content = re.sub(r"!\[.*?\]\(.*?\)", "", content)

            tokens = _tokenize(title + " " + content)
            if not tokens:
                continue

            doc = CorpusDoc(
                doc_id=Path(md_file).stem,
                path=path_str,
                company=company,
                section=section,
                title=title,
                content=content.strip(),
                tokens=tokens,
            )
            self.docs.append(doc)

        print(f"[Retriever] Loaded {len(self.docs)} documents from corpus.")

    # ------------------------------------------------------------------ #
    # Index building                                                       #
    # ------------------------------------------------------------------ #

    def _build_indices(self):
        # Global index
        all_tokens = [d.tokens for d in self.docs]
        self._global_bm25 = BM25Okapi(all_tokens)

        # Per-company indices
        for company in ("hackerrank", "claude", "visa"):
            company_docs = [d for d in self.docs if d.company == company]
            if company_docs:
                tokens = [d.tokens for d in company_docs]
                self._company_index[company] = (BM25Okapi(tokens), company_docs)

        print(f"[Retriever] Built indices for: {list(self._company_index.keys())}")

    # ------------------------------------------------------------------ #
    # Search                                                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        company: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[CorpusDoc, float]]:
        """
        Returns [(doc, score), ...] sorted descending by score.
        If company is given, searches company-scoped index first;
        falls back to global if top score is below threshold.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        results = []

        if company and company.lower() in self._company_index:
            results = self._scoped_search(query_tokens, company.lower(), top_k)
            # Fallback to global if confidence low
            if not results or results[0][1] < self.MIN_SCORE:
                global_results = self._global_search(query_tokens, top_k)
                # Merge: prefer scoped results but supplement with global
                seen = {r[0].doc_id for r in results}
                for gdoc, gscore in global_results:
                    if gdoc.doc_id not in seen:
                        results.append((gdoc, gscore * 0.8))  # slight penalty
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[:top_k]
        else:
            results = self._global_search(query_tokens, top_k)

        return results

    def _scoped_search(
        self, query_tokens: List[str], company: str, top_k: int
    ) -> List[Tuple[CorpusDoc, float]]:
        bm25, docs = self._company_index[company]
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:top_k] if score > 0]

    def _global_search(
        self, query_tokens: List[str], top_k: int
    ) -> List[Tuple[CorpusDoc, float]]:
        scores = self._global_bm25.get_scores(query_tokens)
        ranked = sorted(zip(self.docs, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in ranked[:top_k] if score > 0]

    def get_confidence(self, results: List[Tuple[CorpusDoc, float]]) -> float:
        """Returns the top retrieval score, 0.0 if no results."""
        if not results:
            return 0.0
        return results[0][1]

    def format_context(self, results: List[Tuple[CorpusDoc, float]], max_chars: int = 6000) -> str:
        """Format retrieved docs into a compact context string for the LLM."""
        parts = []
        total = 0
        for i, (doc, score) in enumerate(results):
            header = f"[DOC {i+1}] {doc.title} (section: {doc.section}, company: {doc.company})\n"
            # Truncate long docs
            snippet = doc.content[:2000] if len(doc.content) > 2000 else doc.content
            block = header + snippet + "\n\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "".join(parts)
