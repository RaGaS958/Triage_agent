"""
safety.py — Adversarial detection, risk classification, and escalation triggers.

This module catches:
  1. Prompt injection attempts (asking agent to reveal internals)
  2. Malicious/harmful requests (delete files, hacking, etc.)
  3. High-risk domains requiring human escalation:
     - Fraud / identity theft
     - Billing disputes / refund demands
     - Security vulnerabilities / bug bounty
     - Account access restoration (no admin override possible)
     - Score manipulation requests
     - Legal/compliance issues
  4. Non-English input (log, normalize, still process)
  5. Vague/undecipherable issues
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class RiskLevel(str, Enum):
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"   # always escalate


class EscalationReason(str, Enum):
    BILLING_DISPUTE = "billing_dispute"
    FRAUD_IDENTITY_THEFT = "fraud_identity_theft"
    SECURITY_VULNERABILITY = "security_vulnerability"
    ACCOUNT_ACCESS_RESTORATION = "account_access_restoration"
    SCORE_MANIPULATION = "score_manipulation"
    PROMPT_INJECTION = "prompt_injection"
    MALICIOUS_REQUEST = "malicious_request"
    EXTERNAL_PARTY_ACTION = "external_party_action"   # agent can't act on behalf of 3rd party
    VAGUE_UNDECIPHERABLE = "vague_undecipherable"
    UNSUPPORTED_BY_CORPUS = "unsupported_by_corpus"
    SUBSCRIPTION_CHANGE = "subscription_change"
    EMERGENCY_FINANCIAL = "emergency_financial"
    INFOSEC_COMPLIANCE = "infosec_compliance"
    NONE = "none"


@dataclass
class SafetyAssessment:
    risk_level: RiskLevel
    escalation_reason: EscalationReason
    is_adversarial: bool
    is_malicious: bool
    is_multilingual: bool
    detected_language: str
    flags: list
    notes: str


# ------------------------------------------------------------------ #
# Pattern libraries                                                    #
# ------------------------------------------------------------------ #

PROMPT_INJECTION_PATTERNS = [
    # Asking for system internals
    r"(show|display|reveal|print|list|output|dump|expose|give me|tell me).{0,40}(internal|rules|document|retriev|logic|system|prompt|config|instruction)",
    r"(affiche|montre|révèle|donne.moi).{0,40}(règles|document|interne|logique|système)",  # French
    r"ignore.{0,20}(previous|prior|above|earlier|instruction|rule)",
    r"(act as|pretend to be|you are now|forget|override|bypass|jailbreak)",
    r"(repeat|echo|copy).{0,20}(prompt|instruction|system|context)",
    r"(what|show).{0,20}(system prompt|initial instruction|base prompt)",
]

MALICIOUS_PATTERNS = [
    r"rm\s+-[rf]+\s+[/\*~]",                                              # rm -rf / or rm -r ~
    r"\b(drop\s+table|truncate\s+table|wipe\s+(all|disk|system))\b",
    r"(delete|remove|wipe)\s+(all\s+)?(files?|data|records?)\s+(from\s+)?(the\s+)?system",
    r"\b(hack|exploit|sql\s+injection|xss|csrf|rce|reverse\s+shell)\b",
    r"\b(malware|ransomware|virus|trojan|keylogger|spyware)\b",
    r"(give me|provide|write|create).{0,30}(exploit|payload|shellcode)",
    r"(code|script).{0,20}(delete all|remove all|destroy|wipe)",
]

FRAUD_IDENTITY_PATTERNS = [
    r"\b(identity.{0,5}(theft|stolen|fraud))\b",
    r"\b(my.{0,10}identity.{0,10}(stolen|hacked|compromised))\b",
    r"\b(someone.{0,10}(stole|using|pretending).{0,10}(my|identity|account))\b",
    r"\b(unauthorized.{0,10}(transaction|charge|access|purchase))\b",
    r"\b(fraud(ulent)?|scam|phishing|suspicious.{0,10}(transaction|charge))\b",
]

BILLING_DISPUTE_PATTERNS = [
    r"\b(refund|chargeback|dispute.{0,10}charge|wrong.{0,10}charged)\b",
    r"\b(money.{0,10}back|get.{0,10}refund|want.{0,10}refund)\b",
    r"\b(billing.{0,10}(issue|error|problem|dispute))\b",
    r"\b(order.{0,10}id|payment.{0,10}id|cs_live|transaction.{0,10}id)\b",
    r"\b(overcharged|double.{0,5}charged|incorrect.{0,5}charge)\b",
    r"\b(merchant.{0,20}(wrong product|ignoring|fraud))\b",
]

SECURITY_VULN_PATTERNS = [
    r"\b(security.{0,10}(vulnerability|bug|flaw|exploit|issue))\b",
    r"\b(bug.{0,5}bounty)\b",
    r"\b(found.{0,20}(vulnerability|security.{0,5}issue|bug.{0,5}in))\b",
    r"\b(responsible.{0,10}disclosure|CVE|zero.{0,5}day)\b",
]

ACCOUNT_ACCESS_PATTERNS = [
    r"\b(restore.{0,20}access|give.{0,10}me.{0,10}access|regain.{0,10}access)\b",
    r"\b(lost.{0,10}access|cannot.{0,10}(login|access).{0,10}(my|the|account))\b",
    r"\b(not.{0,10}(the|an).{0,10}(admin|owner|workspace owner))\b",
]

SCORE_MANIPULATION_PATTERNS = [
    r"\b(change|increase|update|fix|adjust).{0,20}(score|result|grade|marks)\b",
    r"\b(unfair(ly)?.{0,20}(graded|scored|assessed|rejected))\b",
    r"\b(move.{0,20}(next round|forward|pass))\b",
    r"\b(platform.{0,10}(wrong|error|bug).{0,10}(my|the).{0,10}(score|result))\b",
]

SUBSCRIPTION_PATTERNS = [
    r"\b(pause|suspend|hold|freeze).{0,20}(subscription|account|plan|billing)\b",
    r"\b(cancel.{0,20}subscription)\b",
]

EMERGENCY_FINANCIAL_PATTERNS = [
    r"\b(urgent.{0,10}cash|need.{0,10}(cash|money).{0,5}(immediately|now|urgently|asap))\b",
    r"\b(emergency.{0,10}(cash|fund|money))\b",
]

INFOSEC_PATTERNS = [
    r"\b(fill.{0,20}(infosec|security|compliance).{0,20}form)\b",
    r"\b(infosec.{0,10}(process|form|questionnaire|audit))\b",
    r"\b(security.{0,10}(audit|questionnaire|assessment|compliance))\b",
    r"\b(vendor.{0,10}security.{0,10}review)\b",
]

NON_LATIN_PATTERN = re.compile(r"[^\x00-\x7F\u00C0-\u024F]")


def _matches_any(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)


def detect_language_simple(text: str) -> str:
    """
    Lightweight heuristic language detection.
    Returns ISO 639-1 code or 'en'.
    """
    # Check for French characters/words
    french_indicators = ["bonjour", "carte", "bloquée", "règles", "affiche", "pendant", "voyage"]
    if any(w in text.lower() for w in french_indicators):
        return "fr"
    # Check for non-ASCII Latin (accented chars common in European langs)
    if NON_LATIN_PATTERN.search(text):
        return "non-english"
    return "en"


def assess_safety(issue: str, subject: str = "", company: str = "") -> SafetyAssessment:
    """
    Full safety and risk assessment for a support ticket.
    Returns a SafetyAssessment with risk_level and escalation_reason.
    """
    combined = f"{subject} {issue}".strip()
    flags = []
    is_adversarial = False
    is_malicious = False

    # --- Prompt injection check ---
    if _matches_any(combined, PROMPT_INJECTION_PATTERNS):
        is_adversarial = True
        flags.append("prompt_injection_detected")

    # --- Malicious intent check ---
    if _matches_any(combined, MALICIOUS_PATTERNS):
        is_malicious = True
        flags.append("malicious_intent_detected")

    # Return CRITICAL immediately for adversarial/malicious
    if is_adversarial or is_malicious:
        lang = detect_language_simple(combined)
        reason = EscalationReason.PROMPT_INJECTION if is_adversarial else EscalationReason.MALICIOUS_REQUEST
        return SafetyAssessment(
            risk_level=RiskLevel.CRITICAL,
            escalation_reason=reason,
            is_adversarial=is_adversarial,
            is_malicious=is_malicious,
            is_multilingual=lang != "en",
            detected_language=lang,
            flags=flags,
            notes=f"Request flagged as {'adversarial (prompt injection)' if is_adversarial else 'malicious'} — immediate escalation required.",
        )

    # --- High-risk domain checks ---
    escalation_reason = EscalationReason.NONE
    risk_level = RiskLevel.LOW

    if _matches_any(combined, FRAUD_IDENTITY_PATTERNS):
        flags.append("fraud_or_identity_theft")
        escalation_reason = EscalationReason.FRAUD_IDENTITY_THEFT
        risk_level = RiskLevel.HIGH

    elif _matches_any(combined, SECURITY_VULN_PATTERNS):
        flags.append("security_vulnerability")
        escalation_reason = EscalationReason.SECURITY_VULNERABILITY
        risk_level = RiskLevel.HIGH

    elif _matches_any(combined, SCORE_MANIPULATION_PATTERNS):
        flags.append("score_manipulation_request")
        escalation_reason = EscalationReason.SCORE_MANIPULATION
        risk_level = RiskLevel.HIGH

    elif _matches_any(combined, BILLING_DISPUTE_PATTERNS):
        flags.append("billing_dispute")
        escalation_reason = EscalationReason.BILLING_DISPUTE
        risk_level = RiskLevel.MEDIUM

    elif _matches_any(combined, ACCOUNT_ACCESS_PATTERNS):
        flags.append("account_access_restoration")
        escalation_reason = EscalationReason.ACCOUNT_ACCESS_RESTORATION
        risk_level = RiskLevel.MEDIUM

    elif _matches_any(combined, EMERGENCY_FINANCIAL_PATTERNS):
        flags.append("emergency_financial_request")
        escalation_reason = EscalationReason.EMERGENCY_FINANCIAL
        risk_level = RiskLevel.HIGH

    elif _matches_any(combined, SUBSCRIPTION_PATTERNS):
        flags.append("subscription_change_request")
        escalation_reason = EscalationReason.SUBSCRIPTION_CHANGE
        risk_level = RiskLevel.MEDIUM

    elif _matches_any(combined, INFOSEC_PATTERNS):
        flags.append("infosec_compliance_request")
        escalation_reason = EscalationReason.INFOSEC_COMPLIANCE
        risk_level = RiskLevel.MEDIUM

    # --- Language detection ---
    lang = detect_language_simple(combined)
    is_multilingual = lang != "en"
    if is_multilingual:
        flags.append(f"non_english_language:{lang}")

    # --- Vagueness check ---
    word_count = len(combined.split())
    if word_count < 5 and not flags:
        flags.append("vague_issue")
        risk_level = RiskLevel.MEDIUM
        escalation_reason = EscalationReason.VAGUE_UNDECIPHERABLE

    notes = ", ".join(flags) if flags else "No risk flags."

    return SafetyAssessment(
        risk_level=risk_level,
        escalation_reason=escalation_reason,
        is_adversarial=is_adversarial,
        is_malicious=is_malicious,
        is_multilingual=is_multilingual,
        detected_language=lang,
        flags=flags,
        notes=notes,
    )


def should_escalate_by_risk(assessment: SafetyAssessment) -> Tuple[bool, str]:
    """
    Returns (should_escalate, reason_str).
    Purely risk-based; corpus confidence check is done in agent.py.
    """
    if assessment.risk_level == RiskLevel.CRITICAL:
        return True, assessment.notes

    HIGH_ESCALATE = {
        EscalationReason.FRAUD_IDENTITY_THEFT,
        EscalationReason.SECURITY_VULNERABILITY,
        EscalationReason.SCORE_MANIPULATION,
        EscalationReason.EMERGENCY_FINANCIAL,
        EscalationReason.MALICIOUS_REQUEST,
        EscalationReason.PROMPT_INJECTION,
    }

    if assessment.escalation_reason in HIGH_ESCALATE:
        return True, f"High-risk flag: {assessment.escalation_reason.value}"

    return False, ""
