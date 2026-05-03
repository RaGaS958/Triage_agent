"""
main.py — Terminal-based support triage agent entry point.

Usage:
    python main.py                          # Process all tickets → output.csv
    python main.py --ticket "your issue"    # Process a single ticket interactively
    python main.py --verbose                # Show detailed per-ticket reasoning

Environment variables required:
    ANTHROPIC_API_KEY   — Anthropic API key

Outputs:
    support_tickets/output.csv  — Predictions for all tickets
    ~/hackerrank_orchestrate/log.txt  — Chat transcript (as required by AGENTS.md)
"""

import os
import sys
import csv
import time
import argparse
import pathlib
import datetime
from typing import Optional

# Ensure code/ is on the path when run from repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agent import TriageAgent, TriageResult
from safety import assess_safety


# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data"
TICKETS_DIR = REPO_ROOT / "support_tickets"
INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"
LOG_DIR = pathlib.Path.home() / "hackerrank_orchestrate"
LOG_FILE = LOG_DIR / "log.txt"


# ------------------------------------------------------------------ #
# Logging                                                             #
# ------------------------------------------------------------------ #

def _ensure_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG_FILE.exists():
        LOG_FILE.touch()


def _ts() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def log_session_start():
    _ensure_log()
    entry = f"""
## [{_ts()}] SESSION START

Agent: multi-domain-support-triage-agent
Repo Root: {REPO_ROOT}
Branch: main
Worktree: main
Parent Agent: none
Language: py
Time Remaining: hackathon ended — running final predictions
"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def log_turn(title: str, user_prompt: str, summary: str, actions: list):
    _ensure_log()
    actions_str = "\n".join(f"* {a}" for a in actions)
    entry = f"""
## [{_ts()}] {title}

User Prompt (verbatim, secrets redacted):
{user_prompt}

Agent Response Summary:
{summary}

Actions:
{actions_str}

Context:
tool=multi-domain-support-triage-agent
branch=main
repo_root={REPO_ROOT}
worktree=main
parent_agent=none
"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


# ------------------------------------------------------------------ #
# Terminal UI                                                          #
# ------------------------------------------------------------------ #

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"


def banner():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════╗
║       Multi-Domain Support Triage Agent  v1.0                ║
║       HackerRank | Claude | Visa                             ║
╚══════════════════════════════════════════════════════════════╝{RESET}
""")


def status_badge(status: str) -> str:
    if status == "replied":
        return f"{GREEN}✓ REPLIED{RESET}"
    return f"{YELLOW}⚡ ESCALATED{RESET}"


def print_result(i: int, issue_preview: str, result: TriageResult, verbose: bool):
    badge = status_badge(result.status)
    print(f"  {BOLD}#{i:02d}{RESET}  {badge}  {DIM}{issue_preview[:60]}...{RESET}")
    if verbose:
        print(f"       {CYAN}Area:{RESET}     {result.product_area}")
        print(f"       {CYAN}Type:{RESET}     {result.request_type}")
        print(f"       {CYAN}Response:{RESET} {result.response[:120]}...")
        print(f"       {CYAN}Why:{RESET}      {result.justification}")
        print()


# ------------------------------------------------------------------ #
# CSV processing                                                       #
# ------------------------------------------------------------------ #

OUTPUT_FIELDS = ["Issue", "Subject", "Company", "status", "product_area", "response", "justification", "request_type"]


def load_tickets(path: pathlib.Path) -> list[dict]:
    tickets = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(dict(row))
    return tickets


def run_batch(agent: TriageAgent, tickets: list[dict], verbose: bool = False):
    """Process all tickets and write output.csv."""
    results = []
    total = len(tickets)

    print(f"\n{BOLD}Processing {total} tickets...{RESET}\n")

    for i, ticket in enumerate(tickets, 1):
        issue = ticket.get("Issue", ticket.get("issue", ""))
        subject = ticket.get("Subject", ticket.get("subject", ""))
        company = ticket.get("Company", ticket.get("company", "None"))

        try:
            result = agent.process(issue=issue, subject=subject, company=company)
        except Exception as e:
            # Hard fallback — never crash
            result = TriageResult(
                status="escalated",
                product_area="general_support",
                response="Your request has been escalated to our support team.",
                justification=f"Agent error during processing: {type(e).__name__}. Escalated for safety.",
                request_type="product_issue",
            )

        results.append({
            "Issue": issue,
            "Subject": subject,
            "Company": company,
            "status": result.status,
            "product_area": result.product_area,
            "response": result.response,
            "justification": result.justification,
            "request_type": result.request_type,
        })

        print_result(i, issue, result, verbose)

        # Log each ticket processing
        log_turn(
            title=f"Process ticket #{i:02d}: {subject or issue[:40]}",
            user_prompt=f"Company={company} | Subject={subject} | Issue={issue[:200]}",
            summary=f"Triaged ticket. Status={result.status}, Area={result.product_area}, Type={result.request_type}.",
            actions=[f"BM25 retrieval + Claude API inference → {result.status}"],
        )

        # Small delay to avoid rate limits
        time.sleep(0.3)

    # Write output CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{BOLD}{GREEN}✓ Output written to: {OUTPUT_CSV}{RESET}")
    return results


def run_interactive(agent: TriageAgent):
    """Interactive single-ticket mode."""
    print(f"\n{BOLD}Interactive mode — enter a support ticket below.{RESET}")
    print(f"{DIM}Press Ctrl+C to exit.{RESET}\n")

    while True:
        try:
            issue = input(f"{CYAN}Issue:{RESET} ").strip()
            if not issue:
                continue
            subject = input(f"{CYAN}Subject (optional):{RESET} ").strip()
            company = input(f"{CYAN}Company (HackerRank/Claude/Visa/None):{RESET} ").strip() or "None"

            print(f"\n{DIM}Thinking...{RESET}")
            result = agent.process(issue=issue, subject=subject, company=company)

            print(f"\n{BOLD}Result:{RESET}")
            print(f"  Status:       {status_badge(result.status)}")
            print(f"  Product Area: {CYAN}{result.product_area}{RESET}")
            print(f"  Request Type: {result.request_type}")
            print(f"\n  {BOLD}Response:{RESET}")
            print(f"  {result.response}")
            print(f"\n  {BOLD}Justification:{RESET}")
            print(f"  {result.justification}\n")

        except KeyboardInterrupt:
            print(f"\n{DIM}Exiting interactive mode.{RESET}")
            break


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Multi-Domain Support Triage Agent")
    parser.add_argument("--ticket", type=str, help="Single issue text for interactive triage")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output per ticket")
    parser.add_argument("--input", type=str, help="Path to input CSV (default: support_tickets/support_tickets.csv)")
    args = parser.parse_args()

    # Check API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{RED}Error: ANTHROPIC_API_KEY environment variable not set.{RESET}")
        print(f"  Set it with: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    banner()
    log_session_start()

    print(f"{DIM}Loading corpus from {DATA_DIR}...{RESET}")
    agent = TriageAgent(data_dir=str(DATA_DIR))

    if args.ticket:
        # Single ticket mode
        result = agent.process(issue=args.ticket, subject="", company="None")
        print(f"\n{BOLD}Result:{RESET}")
        print(f"  Status:       {status_badge(result.status)}")
        print(f"  Product Area: {result.product_area}")
        print(f"  Request Type: {result.request_type}")
        print(f"  Response:     {result.response}")
        print(f"  Justification: {result.justification}")
    elif sys.stdin.isatty() and not args.input and not INPUT_CSV.exists():
        # No input file found → interactive
        run_interactive(agent)
    else:
        # Batch mode
        input_path = pathlib.Path(args.input) if args.input else INPUT_CSV
        if not input_path.exists():
            print(f"{RED}Error: Input file not found: {input_path}{RESET}")
            sys.exit(1)

        tickets = load_tickets(input_path)
        run_batch(agent, tickets, verbose=args.verbose)

        log_turn(
            title="Batch processing complete",
            user_prompt=f"Process all {len(tickets)} tickets from {input_path}",
            summary=f"Successfully processed {len(tickets)} tickets. Output written to {OUTPUT_CSV}.",
            actions=[f"Wrote {OUTPUT_CSV}", f"Logged to {LOG_FILE}"],
        )


if __name__ == "__main__":
    main()
