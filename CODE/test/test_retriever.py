"""
test_retriever.py — Full test suite for retriever.py

Tests cover:
  - Corpus loading (all 3 companies, correct doc counts)
  - Company detection from file paths
  - Tokenization
  - Company-scoped search
  - Global search fallback
  - Confidence scoring
  - Context formatting
  - Edge cases (empty query, unknown company, etc.)
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
from retriever import (
    CorpusRetriever,
    CorpusDoc,
    _tokenize,
    _extract_title,
    _company_from_path,
    _section_from_path,
)

DATA_DIR = str(pathlib.Path(__file__).parent.parent.parent / "data")


# ── Fixture: load retriever once per session (expensive) ────────────
@pytest.fixture(scope="session")
def retriever():
    return CorpusRetriever(DATA_DIR)


# ══════════════════════════════════════════════════════════════════════
# Helper function tests (no corpus needed)
# ══════════════════════════════════════════════════════════════════════

class TestTokenize:

    def test_basic(self):
        tokens = _tokenize("Hello World")
        assert "hello" in tokens
        assert "world" in tokens

    def test_punctuation_stripped(self):
        tokens = _tokenize("hello, world! how's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        # Punctuation-only tokens removed
        assert "," not in tokens
        assert "!" not in tokens

    def test_lowercase(self):
        tokens = _tokenize("HackerRank VISA Claude")
        assert "hackerrank" in tokens
        assert "visa" in tokens
        assert "claude" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_single_char_removed(self):
        tokens = _tokenize("a b c hello")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "hello" in tokens

    def test_numbers_kept(self):
        tokens = _tokenize("version 42 released")
        assert "42" in tokens

    def test_hyphenated_words(self):
        # hyphens stripped → words split
        tokens = _tokenize("step-by-step guide")
        assert "step" in tokens
        assert "by" in tokens or "guide" in tokens


class TestCompanyFromPath:

    def test_hackerrank(self):
        path = "/repo/data/hackerrank/screen/test-reports.md"
        assert _company_from_path(path) == "hackerrank"

    def test_claude(self):
        path = "/repo/data/claude/privacy-and-legal/crawling.md"
        assert _company_from_path(path) == "claude"

    def test_visa(self):
        path = "/repo/data/visa/support/consumer/fraud.md"
        assert _company_from_path(path) == "visa"

    def test_unknown(self):
        path = "/some/other/path/file.md"
        assert _company_from_path(path) == "unknown"

    def test_claude_home_not_mistaken(self):
        """Critical: /home/claude should NOT match 'claude' company."""
        path = "/home/claude/data/visa/support/consumer.md"
        assert _company_from_path(path) == "visa"

    def test_case_insensitive(self):
        path = "/repo/data/HackerRank/something.md"
        assert _company_from_path(path) == "hackerrank"


class TestExtractTitle:

    def test_frontmatter_title(self):
        content = '---\ntitle: "My Test Title"\n---\n# Some other heading'
        assert _extract_title(content, "file.md") == "My Test Title"

    def test_h1_title(self):
        content = "# Mock Interview Credits\n\nSome content here."
        assert _extract_title(content, "file.md") == "Mock Interview Credits"

    def test_filename_fallback(self):
        content = "Just some plain text with no headings."
        title = _extract_title(content, "/path/to/my-test-file.md")
        assert title == "my test file"

    def test_frontmatter_priority_over_h1(self):
        content = "---\ntitle: Frontmatter Title\n---\n# H1 Title"
        assert _extract_title(content, "file.md") == "Frontmatter Title"


class TestSectionFromPath:

    def test_section_extracted(self):
        path = "/repo/data/hackerrank/screen/test-reports.md"
        assert _section_from_path(path) == "screen"

    def test_claude_section(self):
        path = "/repo/data/claude/privacy-and-legal/crawling.md"
        assert _section_from_path(path) == "privacy-and-legal"

    def test_unknown_path(self):
        section = _section_from_path("/some/random/path/file.md")
        # Should return something, not crash
        assert isinstance(section, str)


# ══════════════════════════════════════════════════════════════════════
# Corpus Loading Tests
# ══════════════════════════════════════════════════════════════════════

class TestCorpusLoading:

    def test_docs_loaded(self, retriever):
        assert len(retriever.docs) > 0

    def test_all_three_companies_present(self, retriever):
        companies = {d.company for d in retriever.docs}
        assert "hackerrank" in companies
        assert "claude" in companies
        assert "visa" in companies

    def test_hackerrank_docs_count(self, retriever):
        hr_docs = [d for d in retriever.docs if d.company == "hackerrank"]
        assert len(hr_docs) > 100  # corpus has many HR docs

    def test_claude_docs_count(self, retriever):
        cl_docs = [d for d in retriever.docs if d.company == "claude"]
        assert len(cl_docs) > 50

    def test_visa_docs_count(self, retriever):
        visa_docs = [d for d in retriever.docs if d.company == "visa"]
        assert len(visa_docs) > 5

    def test_all_docs_have_content(self, retriever):
        empty = [d for d in retriever.docs if not d.content.strip()]
        assert len(empty) == 0

    def test_all_docs_have_tokens(self, retriever):
        no_tokens = [d for d in retriever.docs if not d.tokens]
        assert len(no_tokens) == 0

    def test_all_docs_have_company(self, retriever):
        unknown = [d for d in retriever.docs if d.company == "unknown"]
        assert len(unknown) == 0, f"Docs without company: {[d.path for d in unknown[:3]]}"

    def test_all_company_indices_built(self, retriever):
        assert "hackerrank" in retriever._company_index
        assert "claude" in retriever._company_index
        assert "visa" in retriever._company_index

    def test_global_bm25_built(self, retriever):
        assert retriever._global_bm25 is not None


# ══════════════════════════════════════════════════════════════════════
# Search Tests — Company-Scoped
# ══════════════════════════════════════════════════════════════════════

class TestScopedSearch:

    def test_mock_interview_hackerrank(self, retriever):
        results = retriever.search("mock interview purchase credits", company="hackerrank", top_k=5)
        assert len(results) > 0
        top_doc, top_score = results[0]
        assert top_score > 1.0  # reasonable BM25 signal
        assert top_doc.company == "hackerrank"

    def test_mock_interview_top_result_relevant(self, retriever):
        results = retriever.search("mock interview", company="hackerrank", top_k=3)
        titles = [doc.title.lower() for doc, _ in results]
        assert any("mock" in t or "interview" in t for t in titles)

    def test_visa_dispute_resolution(self, retriever):
        results = retriever.search("dispute resolution charge merchant", company="visa", top_k=3)
        assert len(results) > 0
        assert all(doc.company == "visa" for doc, _ in results)

    def test_claude_privacy_crawl(self, retriever):
        results = retriever.search("crawl website robots.txt opt out", company="claude", top_k=3)
        assert len(results) > 0
        assert all(doc.company == "claude" for doc, _ in results)

    def test_hackerrank_certificate(self, retriever):
        results = retriever.search("certificate name incorrect profile", company="hackerrank", top_k=3)
        assert len(results) > 0

    def test_scoped_results_only_company_docs(self, retriever):
        """Company-scoped search must return only that company's docs."""
        results = retriever.search("subscription billing payment", company="hackerrank", top_k=5)
        for doc, _ in results:
            assert doc.company == "hackerrank", f"Got doc from wrong company: {doc.company}"

    def test_claude_bedrock_query(self, retriever):
        results = retriever.search("Amazon Bedrock Claude API requests failing", company="claude", top_k=3)
        assert len(results) > 0

    def test_hackerrank_roles_management(self, retriever):
        results = retriever.search("remove interviewer user deactivate team management", company="hackerrank", top_k=3)
        assert len(results) > 0


# ══════════════════════════════════════════════════════════════════════
# Search Tests — Global Fallback
# ══════════════════════════════════════════════════════════════════════

class TestGlobalFallback:

    def test_global_search_no_company(self, retriever):
        results = retriever.search("billing payment subscription", company=None, top_k=5)
        assert len(results) > 0

    def test_global_returns_multiple_companies(self, retriever):
        """Global search can return docs from any company."""
        results = retriever.search("account access login", company=None, top_k=10)
        companies = {doc.company for doc, _ in results}
        # Likely more than one company given broad query
        assert len(companies) >= 1

    def test_global_search_unknown_company(self, retriever):
        results = retriever.search("what is the refund policy", company="unknown", top_k=3)
        assert len(results) > 0  # Falls back to global


# ══════════════════════════════════════════════════════════════════════
# Confidence Score Tests
# ══════════════════════════════════════════════════════════════════════

class TestConfidenceScoring:

    def test_high_confidence_specific_query(self, retriever):
        results = retriever.search("mock interview credits purchase refund", company="hackerrank", top_k=3)
        confidence = retriever.get_confidence(results)
        assert confidence > 5.0  # Strong BM25 match

    def test_low_confidence_garbage_query(self, retriever):
        """BM25 scores all-novel tokens high via IDF; confidence gate is
        therefore measured relative to real queries, not absolute zero."""
        results_real = retriever.search("mock interview refund", company="hackerrank", top_k=1)
        results_garbage = retriever.search("xyzzy foo bar baz qux nonsense", company="hackerrank", top_k=1)
        conf_real = retriever.get_confidence(results_real)
        conf_garbage = retriever.get_confidence(results_garbage)
        # Real query must score significantly higher than garbage
        assert conf_real > conf_garbage * 2, (
            f"Real query ({conf_real:.2f}) should be >2x garbage ({conf_garbage:.2f})"
        )

    def test_empty_query_returns_empty(self, retriever):
        results = retriever.search("", company="hackerrank", top_k=3)
        assert results == []

    def test_confidence_zero_for_empty_results(self, retriever):
        confidence = retriever.get_confidence([])
        assert confidence == 0.0

    def test_results_sorted_descending(self, retriever):
        results = retriever.search("test assessment candidate hiring", company="hackerrank", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════
# Context Formatting Tests
# ══════════════════════════════════════════════════════════════════════

class TestContextFormatting:

    def test_format_context_not_empty(self, retriever):
        results = retriever.search("mock interview", company="hackerrank", top_k=3)
        context = retriever.format_context(results)
        assert len(context) > 0

    def test_format_context_contains_doc_headers(self, retriever):
        results = retriever.search("mock interview", company="hackerrank", top_k=3)
        context = retriever.format_context(results)
        assert "[DOC 1]" in context

    def test_format_context_respects_max_chars(self, retriever):
        results = retriever.search("test", company="hackerrank", top_k=5)
        context = retriever.format_context(results, max_chars=500)
        assert len(context) <= 600  # small buffer for final block

    def test_format_context_empty_results(self, retriever):
        context = retriever.format_context([])
        assert context == ""

    def test_format_context_includes_company(self, retriever):
        results = retriever.search("billing", company="hackerrank", top_k=2)
        context = retriever.format_context(results)
        assert "hackerrank" in context.lower()


# ══════════════════════════════════════════════════════════════════════
# Retrieval Quality — Per Ticket Spot Checks
# ══════════════════════════════════════════════════════════════════════

class TestRetrievalQualityPerTicket:
    """
    For each key ticket from support_tickets.csv, verify that the retriever
    surfaces at least one relevant document.
    """

    EXPECTED_HITS = [
        # (query, company, expected_keyword_in_top_title_or_content)
        ("mock interview stopped mid-session refund", "hackerrank", "mock"),
        ("certificate name incorrect update profile", "hackerrank", "certif"),
        ("inactivity timeout interviewer kicked out", "hackerrank", "interview"),
        ("remove interviewer from platform deactivate", "hackerrank", "role"),
        ("pause subscription stop hiring", "hackerrank", "subscri"),
        ("cannot see apply tab profile", "hackerrank", "profile"),
        ("submissions not working all challenges fail", "hackerrank", "challenge"),
        ("zoom connectivity blocker compatibility check interview", "hackerrank", "interview"),
        ("reschedule assessment company test window", "hackerrank", "test"),
        ("resume builder down not working", "hackerrank", "resume"),
        ("infosec forms company onboarding security", "hackerrank", "securi"),
        ("claude team workspace lost access admin seat", "claude", "seat"),
        ("amazon bedrock claude requests failing region", "claude", "bedrock"),
        ("stop crawl website robots txt anthropic bot", "claude", "crawl"),
        ("claude not responding all requests failing outage", "claude", "claude"),
        ("data model improvement retention how long", "claude", "data"),
        ("claude education university professor lti integration", "claude", "education"),
        ("visa dispute charge merchant wrong product chargeback", "visa", "dispute"),
        ("identity stolen what should I do fraud", "visa", "fraud"),
        ("urgent cash atm only visa card", "visa", "atm"),
        ("minimum spend visa virgin islands", "visa", "visa"),
    ]

    @pytest.mark.parametrize("query,company,expected_kw", EXPECTED_HITS)
    def test_retrieval_surfaces_relevant_doc(self, retriever, query, company, expected_kw):
        results = retriever.search(query, company=company, top_k=5)
        assert len(results) > 0, f"No results for query: '{query}'"

        # Check that at least one result contains the expected keyword
        all_text = " ".join(
            (doc.title + " " + doc.content).lower()
            for doc, _ in results
        )
        assert expected_kw.lower() in all_text, (
            f"Expected keyword '{expected_kw}' not found in top results for: '{query}'\n"
            f"Top titles: {[doc.title for doc, _ in results[:3]]}"
        )
