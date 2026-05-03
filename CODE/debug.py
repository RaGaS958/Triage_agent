"""
debug.py — Interactive debug tool for the triage pipeline.

Usage:
    python code/debug.py                        # Menu-driven interactive debug
    python code/debug.py --ticket-id 5          # Debug ticket #5 from support_tickets.csv
    python code/debug.py --issue "your text"    # Debug a custom issue
    python code/debug.py --all                  # Debug all tickets, show summary table
    python code/debug.py --retrieval "query"    # Test retrieval only (no LLM)
    python code/debug.py --safety "text"        # Test safety assessment only

Environment:
    ANTHROPIC_API_KEY  — required for --ticket-id and --issue modes
"""

import os
import sys
import csv
import json
import pathlib
import argparse
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from retriever import CorpusRetriever
from safety import assess_safety, should_escalate_by_risk, RiskLevel

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data"
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"

# ── Terminal colours ────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"


def hr(char="─", width=70):
    print(f"{DIM}{char * width}{RESET}")


def section(title: str):
    hr()
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    hr()


def badge(label: str, value: str, color: str = CYAN):
    print(f"  {BOLD}{color}{label:<20}{RESET} {value}")


# ══════════════════════════════════════════════════════════════════════
# Safety Debug
# ══════════════════════════════════════════════════════════════════════

def debug_safety(issue: str, subject: str = "", company: str = ""):
    section("SAFETY ASSESSMENT")
    a = assess_safety(issue, subject, company)
    should_esc, esc_reason = should_escalate_by_risk(a)

    risk_color = {
        RiskLevel.SAFE: GREEN,
        RiskLevel.LOW: GREEN,
        RiskLevel.MEDIUM: YELLOW,
        RiskLevel.HIGH: RED,
        RiskLevel.CRITICAL: RED,
    }.get(a.risk_level, CYAN)

    badge("Risk Level:", str(a.risk_level.value).upper(), risk_color)
    badge("Escalation Reason:", a.escalation_reason.value)
    badge("Is Adversarial:", str(a.is_adversarial), RED if a.is_adversarial else GREEN)
    badge("Is Malicious:", str(a.is_malicious), RED if a.is_malicious else GREEN)
    badge("Is Multilingual:", str(a.is_multilingual))
    badge("Language:", a.detected_language)
    badge("Should Escalate:", f"{should_esc}  ({esc_reason})", RED if should_esc else GREEN)
    print()
    print(f"  {BOLD}Flags:{RESET}")
    if a.flags:
        for flag in a.flags:
            print(f"    {YELLOW}⚠  {flag}{RESET}")
    else:
        print(f"    {GREEN}✓  No flags{RESET}")
    print()
    return a


# ══════════════════════════════════════════════════════════════════════
# Retrieval Debug
# ══════════════════════════════════════════════════════════════════════

def debug_retrieval(retriever: CorpusRetriever, query: str, company: str = None, top_k: int = 5):
    section(f"BM25 RETRIEVAL — company={company or 'global'}")
    print(f"  Query: {BOLD}{query[:80]}{RESET}\n")

    results = retriever.search(query, company=company, top_k=top_k)
    confidence = retriever.get_confidence(results)

    conf_color = GREEN if confidence >= 0.3 else RED
    badge("Confidence:", f"{confidence:.4f}", conf_color)
    badge("Docs Found:", str(len(results)))
    print()

    if not results:
        print(f"  {RED}No results found.{RESET}")
    else:
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {BOLD}[{i}]{RESET} Score={score:.2f}  Company={CYAN}{doc.company}{RESET}  Section={doc.section}")
            print(f"      Title: {doc.title}")
            print(f"      Path:  {DIM}{doc.path[-80:]}{RESET}")
            # Show content preview
            preview = doc.content[:200].replace("\n", " ")
            print(f"      Preview: {DIM}{preview}...{RESET}")
            print()

    return results, confidence


# ══════════════════════════════════════════════════════════════════════
# Full Pipeline Debug (requires ANTHROPIC_API_KEY)
# ══════════════════════════════════════════════════════════════════════

def debug_full_pipeline(retriever: CorpusRetriever, issue: str, subject: str, company: str):
    print()
    hr("═")
    print(f"{BOLD}{MAGENTA}  FULL TRIAGE PIPELINE DEBUG{RESET}")
    hr("═")
    print(f"  {BOLD}Issue:{RESET}   {issue[:100]}")
    print(f"  {BOLD}Subject:{RESET} {subject[:60] or '(none)'}")
    print(f"  {BOLD}Company:{RESET} {company}")
    print()

    # ── Step 1: Safety ──────────────────────────────────────────────
    section("STEP 1 — SAFETY ASSESSMENT")
    safety = assess_safety(issue, subject, company)
    should_esc_safety, esc_reason = should_escalate_by_risk(safety)

    risk_color = RED if safety.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else YELLOW if safety.risk_level == RiskLevel.MEDIUM else GREEN
    badge("Risk Level:", safety.risk_level.value.upper(), risk_color)
    badge("Adversarial:", str(safety.is_adversarial), RED if safety.is_adversarial else GREEN)
    badge("Malicious:", str(safety.is_malicious), RED if safety.is_malicious else GREEN)
    badge("Escalate (safety):", str(should_esc_safety), RED if should_esc_safety else GREEN)
    badge("Reason:", esc_reason or "None")

    if safety.is_adversarial:
        print(f"\n  {RED}{BOLD}⛔ STOPPED: Prompt injection detected. Pipeline terminated.{RESET}")
        print(f"  No LLM call will be made.")
        return

    if safety.is_malicious:
        print(f"\n  {RED}{BOLD}⛔ STOPPED: Malicious request detected. Pipeline terminated.{RESET}")
        return

    # ── Step 2: Company inference ────────────────────────────────────
    section("STEP 2 — COMPANY INFERENCE")
    from agent import infer_company, COMPANY_KEYWORDS
    effective_company = company
    if company.lower() in ("none", "", "unknown"):
        effective_company = infer_company(issue, subject)
        print(f"  Company was '{company}' → inferred as {BOLD}{CYAN}{effective_company}{RESET}")
    else:
        print(f"  Company provided: {BOLD}{CYAN}{effective_company}{RESET} (no inference needed)")

    # ── Step 3: Retrieval ────────────────────────────────────────────
    query = f"{subject} {issue}".strip()
    results, confidence = debug_retrieval(retriever, query, company=effective_company)

    # ── Step 4: Confidence gate ──────────────────────────────────────
    section("STEP 4 — CONFIDENCE GATE")
    threshold = 0.3
    gate_pass = confidence >= threshold
    gate_color = GREEN if gate_pass else RED

    badge("Confidence:", f"{confidence:.4f} (threshold={threshold})", gate_color)
    badge("Gate:", "PASS — proceeding to LLM" if gate_pass else "FAIL — will escalate", gate_color)

    if not gate_pass:
        print(f"\n  {YELLOW}↳ No relevant corpus docs found. Will hint LLM to escalate.{RESET}")

    # ── Step 5: LLM call ────────────────────────────────────────────
    section("STEP 5 — LLM CALL (Claude Haiku)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"  {YELLOW}ANTHROPIC_API_KEY not set — skipping LLM call.{RESET}")
        print(f"  Set: export ANTHROPIC_API_KEY=your_key")
        return

    print(f"  {DIM}Calling Claude API...{RESET}")
    try:
        from agent import TriageAgent
        agent = TriageAgent(str(DATA_DIR))
        agent.retriever = retriever  # reuse already-loaded retriever
        result = agent.process(issue, subject, company)

        section("STEP 6 — RESULT")
        status_color = GREEN if result.status == "replied" else YELLOW
        badge("Status:", result.status.upper(), status_color)
        badge("Product Area:", result.product_area)
        badge("Request Type:", result.request_type)
        print()
        print(f"  {BOLD}Response:{RESET}")
        for line in textwrap.wrap(result.response, width=70):
            print(f"    {line}")
        print()
        print(f"  {BOLD}Justification:{RESET}")
        for line in textwrap.wrap(result.justification, width=70):
            print(f"    {DIM}{line}{RESET}")
        print()

    except Exception as e:
        print(f"  {RED}Error calling LLM: {e}{RESET}")


# ══════════════════════════════════════════════════════════════════════
# Summary Table (all tickets)
# ══════════════════════════════════════════════════════════════════════

def debug_all_tickets(retriever: CorpusRetriever):
    """Show a summary of output.csv with retrieval confidence for each ticket."""
    if not OUTPUT_CSV.exists():
        print(f"{RED}output.csv not found. Run generate_output.py first.{RESET}")
        return

    input_rows = []
    if INPUT_CSV.exists():
        with open(INPUT_CSV, newline="", encoding="utf-8") as f:
            input_rows = list(csv.DictReader(f))

    output_rows = []
    with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
        output_rows = list(csv.DictReader(f))

    hr("═")
    print(f"{BOLD}{MAGENTA}  ALL TICKETS — PREDICTION SUMMARY{RESET}")
    hr("═")
    print(f"  {'#':<4} {'STATUS':<12} {'TYPE':<17} {'CONF':<8} {'AREA':<30} ISSUE")
    hr()

    replied = escalated = 0
    for i, row in enumerate(output_rows, 1):
        issue = row.get("Issue", "")
        subject = row.get("Subject", "")
        company = row.get("Company", "")
        status = row.get("status", "?")
        area = row.get("product_area", "?")
        req_type = row.get("request_type", "?")

        # Get retrieval confidence for info
        query = f"{subject} {issue}".strip()
        results = retriever.search(query, company=company.lower() if company.lower() not in ("none", "") else None, top_k=1)
        conf = retriever.get_confidence(results)
        conf_str = f"{conf:.1f}"

        status_color = GREEN if status == "replied" else YELLOW
        conf_color = GREEN if conf >= 5.0 else YELLOW if conf >= 0.3 else RED

        print(
            f"  {i:<4} "
            f"{status_color}{status.upper():<12}{RESET} "
            f"{req_type:<17} "
            f"{conf_color}{conf_str:<8}{RESET} "
            f"{area[:30]:<30} "
            f"{DIM}{issue[:40]}...{RESET}"
        )
        if status == "replied":
            replied += 1
        else:
            escalated += 1

    hr()
    print(f"  {BOLD}Total:{RESET} {len(output_rows)} tickets  |  "
          f"{GREEN}Replied: {replied}{RESET}  |  "
          f"{YELLOW}Escalated: {escalated}{RESET}")
    print()


# ══════════════════════════════════════════════════════════════════════
# Interactive Menu
# ══════════════════════════════════════════════════════════════════════

def interactive_menu(retriever: CorpusRetriever):
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   Support Triage Agent — Debug Console       ║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════╝{RESET}")
    print()
    print(f"  {BOLD}1){RESET}  Debug a specific ticket by ID")
    print(f"  {BOLD}2){RESET}  Debug a custom issue")
    print(f"  {BOLD}3){RESET}  Test retrieval only")
    print(f"  {BOLD}4){RESET}  Test safety assessment only")
    print(f"  {BOLD}5){RESET}  Show all ticket predictions")
    print(f"  {BOLD}q){RESET}  Quit")
    print()

    while True:
        choice = input(f"{CYAN}→ {RESET}").strip().lower()

        if choice == "q":
            break

        elif choice == "1":
            ticket_id = input("Ticket ID (1-29): ").strip()
            if INPUT_CSV.exists():
                with open(INPUT_CSV, newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                try:
                    idx = int(ticket_id) - 1
                    row = rows[idx]
                    debug_full_pipeline(
                        retriever,
                        row.get("Issue", ""),
                        row.get("Subject", ""),
                        row.get("Company", "None"),
                    )
                except (ValueError, IndexError):
                    print(f"{RED}Invalid ticket ID.{RESET}")
            else:
                print(f"{RED}support_tickets.csv not found.{RESET}")

        elif choice == "2":
            issue = input("Issue text: ").strip()
            subject = input("Subject (optional): ").strip()
            company = input("Company (HackerRank/Claude/Visa/None): ").strip() or "None"
            debug_full_pipeline(retriever, issue, subject, company)

        elif choice == "3":
            query = input("Search query: ").strip()
            company = input("Company (or leave blank for global): ").strip() or None
            debug_retrieval(retriever, query, company=company)

        elif choice == "4":
            issue = input("Issue text: ").strip()
            subject = input("Subject: ").strip()
            company = input("Company: ").strip()
            debug_safety(issue, subject, company)

        elif choice == "5":
            debug_all_tickets(retriever)

        else:
            print(f"{RED}Unknown option.{RESET}")

        print()


# ══════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Triage Agent Debug Console")
    parser.add_argument("--ticket-id", type=int, help="Debug ticket N from support_tickets.csv")
    parser.add_argument("--issue", type=str, help="Debug a custom issue text")
    parser.add_argument("--subject", type=str, default="", help="Subject for --issue mode")
    parser.add_argument("--company", type=str, default="None", help="Company for --issue mode")
    parser.add_argument("--retrieval", type=str, help="Test retrieval only for this query")
    parser.add_argument("--retrieval-company", type=str, default=None, help="Company scope for retrieval test")
    parser.add_argument("--safety", type=str, help="Test safety assessment only for this text")
    parser.add_argument("--all", action="store_true", help="Show full summary of all tickets")
    args = parser.parse_args()

    print(f"{DIM}Loading corpus...{RESET}")
    retriever = CorpusRetriever(str(DATA_DIR))

    if args.safety:
        debug_safety(args.safety, "", "")

    elif args.retrieval:
        debug_retrieval(retriever, args.retrieval, company=args.retrieval_company)

    elif args.all:
        debug_all_tickets(retriever)

    elif args.ticket_id:
        if not INPUT_CSV.exists():
            print(f"{RED}support_tickets.csv not found at {INPUT_CSV}{RESET}")
            sys.exit(1)
        with open(INPUT_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        try:
            row = rows[args.ticket_id - 1]
        except IndexError:
            print(f"{RED}Ticket ID {args.ticket_id} out of range (1-{len(rows)}){RESET}")
            sys.exit(1)
        debug_full_pipeline(
            retriever,
            row.get("Issue", ""),
            row.get("Subject", ""),
            row.get("Company", "None"),
        )

    elif args.issue:
        debug_full_pipeline(retriever, args.issue, args.subject, args.company)

    else:
        interactive_menu(retriever)


if __name__ == "__main__":
    main()
