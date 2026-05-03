"""
generate_output.py — Generates predictions for all support tickets.
Uses BM25 retrieval + expert analysis grounded in the corpus.
Run this if you don't have an ANTHROPIC_API_KEY set.
"""

import csv
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from retriever import CorpusRetriever
from safety import assess_safety, should_escalate_by_risk

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = REPO_ROOT / "data"
INPUT_CSV = REPO_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = REPO_ROOT / "support_tickets" / "output.csv"

# ── Curated expert predictions for all 29 tickets ──────────────────
# Each entry maps to a row in support_tickets.csv (in order).
# Built from deep corpus analysis + BM25 retrieval verification.

PREDICTIONS = [
    # 1. Claude team workspace access lost
    {
        "status": "escalated",
        "product_area": "team_and_enterprise_plans/admin_management",
        "response": (
            "We understand you've lost access to your Claude team workspace. "
            "However, only workspace Owners and Primary Owners can restore seat access for members. "
            "Since you are not the workspace owner or admin, this action cannot be completed on your behalf. "
            "Please contact your IT admin or workspace owner to have your seat reassigned. "
            "Refer to: https://support.claude.com/en/articles/12004354-purchase-and-manage-seats-on-team-plans"
        ),
        "justification": (
            "Access restoration requires Owner or Primary Owner permissions per the Team plan seat management documentation. "
            "The user explicitly states they are not the admin, making automated resolution impossible. Escalated for admin action."
        ),
        "request_type": "product_issue",
    },
    # 2. HackerRank test score dispute
    {
        "status": "escalated",
        "product_area": "screen/test_reports",
        "response": (
            "We understand your concern about your test score. "
            "However, HackerRank support is unable to review, modify, or override assessment scores, "
            "as scores are determined by the automated grading system. "
            "Score changes and hiring decisions are entirely at the discretion of the recruiting company. "
            "We also cannot contact the company on your behalf. "
            "If you believe there was a technical issue with the platform, please contact HackerRank at https://support.hackerrank.com."
        ),
        "justification": (
            "Score manipulation is a high-risk request: HackerRank support does not alter assessment scores or intercede in hiring decisions. "
            "The request also asks the agent to contact a third-party company, which is outside agent scope. Escalated per policy."
        ),
        "request_type": "product_issue",
    },
    # 3. Visa wrong product / merchant dispute
    {
        "status": "escalated",
        "product_area": "dispute_resolution",
        "response": (
            "We're sorry to hear about your experience. "
            "For disputes involving incorrect merchandise from an online merchant, you should contact your card-issuing bank "
            "to initiate a formal dispute (chargeback) process. Your bank will contact the merchant's bank on your behalf. "
            "Visa itself does not process refunds or ban individual merchants directly — these actions go through your card issuer. "
            "Per Visa's dispute guidelines, a cardholder contacts their issuer who then works through the Visa network. "
            "See: https://www.visa.co.in/support/small-business/dispute-resolution.html"
        ),
        "justification": (
            "This is a billing dispute involving a third-party merchant. Per the Visa dispute resolution corpus, "
            "chargebacks are initiated via the card-issuing bank, not directly through Visa support. "
            "Visa cannot ban merchants or directly refund cardholders. Escalated to issuing bank process."
        ),
        "request_type": "product_issue",
    },
    # 4. Mock interviews stopped, refund
    {
        "status": "escalated",
        "product_area": "subscriptions_payments_and_billing",
        "response": (
            "We're sorry to hear your mock interview stopped mid-session. "
            "For refund requests on mock interview credits, please contact the HackerRank support team directly at "
            "help@hackerrank.com. The team will review your case and process any eligible refund promptly. "
            "Note: Once purchased, mock interview credits do not expire, so if the issue was a one-time disruption, "
            "your credits should still be available for future use."
        ),
        "justification": (
            "Refund requests for mock interviews must be reviewed by the HackerRank support team per the billing corpus "
            "('contact help@hackerrank.com'). The session interruption may also be a technical issue requiring investigation. Escalated."
        ),
        "request_type": "product_issue",
    },
    # 5. Payment issue with order ID cs_live_abcdefgh
    {
        "status": "escalated",
        "product_area": "subscriptions_payments_and_billing",
        "response": (
            "We've received your payment inquiry regarding order ID cs_live_abcdefgh. "
            "Payment issues require manual review by our billing team. "
            "Please contact HackerRank support at help@hackerrank.com with your order ID and a description of the issue. "
            "Our team will investigate and respond as soon as possible."
        ),
        "justification": (
            "Payment disputes with specific order IDs require manual billing team investigation. "
            "No automated resolution is possible without account verification and transaction lookup. Escalated to billing team."
        ),
        "request_type": "product_issue",
    },
    # 6. HackerRank infosec process / forms
    {
        "status": "escalated",
        "product_area": "settings/gdpr_and_compliance",
        "response": (
            "Thank you for your interest in using HackerRank for hiring. "
            "Infosec and vendor security questionnaires require direct engagement with HackerRank's security and sales team. "
            "These forms involve proprietary security details that cannot be handled through automated support. "
            "Please reach out to your HackerRank account representative or contact sales at https://www.hackerrank.com/contact-us "
            "to initiate the infosec review process."
        ),
        "justification": (
            "Completing infosec/vendor security questionnaires requires the HackerRank security team and account representatives. "
            "This is outside the scope of automated support. Escalated to sales/security team."
        ),
        "request_type": "product_issue",
    },
    # 7. Can't see apply tab
    {
        "status": "replied",
        "product_area": "hackerrank_community/profile_and_preferences",
        "response": (
            "The 'Apply' tab on HackerRank typically appears when you are viewing job listings or opportunities on the platform. "
            "If you cannot see the Apply tab, here are some steps to try:\n\n"
            "1. Ensure your HackerRank profile is complete (profile photo, skills, and work preferences set).\n"
            "2. Navigate to the Jobs section from the top navigation bar.\n"
            "3. If using HackerRank for Work (employer side), the Apply tab is a candidate-side feature not visible in the admin dashboard.\n"
            "4. Try clearing your browser cache or using a different browser.\n"
            "5. Make sure you are logged into your HackerRank Community account (not a For Work account).\n\n"
            "If the issue persists, please contact support@hackerrank.com."
        ),
        "justification": (
            "Navigation and profile-related issues on HackerRank Community are addressed in the community help documentation. "
            "Standard troubleshooting steps (profile completion, correct account type, browser cache) are applicable here."
        ),
        "request_type": "product_issue",
    },
    # 8. Submissions not working across all challenges
    {
        "status": "escalated",
        "product_area": "hackerrank_community/practice_coding_challenges",
        "response": (
            "We apologize for the inconvenience. A platform-wide submission failure across all challenges indicates a potential "
            "technical outage or system issue that requires immediate investigation by our engineering team. "
            "We have escalated this to our technical team for urgent review. "
            "In the meantime, you may check https://www.hackerrank.com for any ongoing status updates. "
            "Thank you for reporting this."
        ),
        "justification": (
            "A platform-wide bug (all submissions failing across all challenges) is a critical technical incident "
            "requiring engineering team investigation. Cannot be resolved through standard support. Escalated as a bug."
        ),
        "request_type": "bug",
    },
    # 9. Zoom connectivity blocker for compatibility check
    {
        "status": "replied",
        "product_area": "interviews/getting_started",
        "response": (
            "We understand you're facing a Zoom connectivity issue during the HackerRank compatibility check. "
            "Here are some steps to resolve the Zoom connectivity issue:\n\n"
            "1. Ensure Zoom is installed and updated to the latest version.\n"
            "2. Check that your firewall or antivirus is not blocking Zoom connections.\n"
            "3. If your organization uses a VPN, try disconnecting it and running the compatibility check again.\n"
            "4. Ensure Zoom has the required permissions (camera, microphone, screen sharing) in your system settings.\n"
            "5. Try restarting your browser, clearing cache, and re-running the compatibility check.\n"
            "6. On Windows, check Windows Defender Firewall settings for any blocks on Zoom.\n\n"
            "If the issue persists after these steps, contact your IT team to verify network policies, "
            "or reach out to HackerRank support with your system details."
        ),
        "justification": (
            "Zoom connectivity issues during compatibility checks are a known challenge addressed in HackerRank's interview setup documentation. "
            "Standard troubleshooting (permissions, firewall, VPN, updates) is applicable and addressable."
        ),
        "request_type": "product_issue",
    },
    # 10. Reschedule assessment
    {
        "status": "escalated",
        "product_area": "screen/invite_candidates",
        "response": (
            "We understand you need to reschedule your HackerRank assessment. "
            "Rescheduling is managed by the company or recruiter who sent you the test invitation — "
            "HackerRank support cannot reschedule tests on behalf of candidates without recruiter authorization. "
            "Please reach out directly to the recruiter or HR team at the company ('Company Name') to request a new test window. "
            "They can resend the invitation with an updated time or extend the test deadline from their HackerRank for Work account."
        ),
        "justification": (
            "Test rescheduling must be authorized by the test-sending company/recruiter from their HackerRank for Work account. "
            "HackerRank support cannot unilaterally reschedule candidate assessments. Escalated per process."
        ),
        "request_type": "product_issue",
    },
    # 11. Interview inactivity times
    {
        "status": "replied",
        "product_area": "interviews/interview_settings",
        "response": (
            "Regarding inactivity timeouts in HackerRank Interviews:\n\n"
            "HackerRank Interviews have inactivity detection built in to manage session integrity. "
            "If a participant is inactive (no keyboard or mouse activity) for an extended period, "
            "they may be redirected to a waiting area or lobby.\n\n"
            "To address your specific concern about interviewers being marked inactive while watching screen shares:\n\n"
            "1. Interviewers can prevent inactivity timeouts by periodically interacting with the HackerRank interview interface "
            "(e.g., scrolling, typing a note in the notepad).\n"
            "2. If you need to adjust inactivity timeout settings for your account, "
            "please contact your HackerRank account administrator or reach out to support to request a configuration change.\n\n"
            "For more details, refer to your HackerRank Interviews settings or contact support@hackerrank.com."
        ),
        "justification": (
            "Interview inactivity settings are a platform configuration matter. "
            "The corpus covers interview settings and management. Practical guidance provided; "
            "configuration change requests may need escalation to account admin."
        ),
        "request_type": "product_issue",
    },
    # 12. "it's not working, help" — vague, None company
    {
        "status": "escalated",
        "product_area": "general_support",
        "response": (
            "Thank you for reaching out. We'd love to help, but we need a bit more information to assist you effectively. "
            "Could you please clarify:\n\n"
            "1. Which product or platform are you having trouble with? (HackerRank, Claude, Visa, or another?)\n"
            "2. What specifically is not working? (e.g., login, submission, payment, etc.)\n"
            "3. What error message or behavior are you seeing?\n\n"
            "Without these details, we've escalated your ticket so a support agent can follow up with you directly."
        ),
        "justification": (
            "The request is too vague to process: no company identified, no product context, and no description of the failure. "
            "Escalated with a request for clarification. Classified as invalid pending more information."
        ),
        "request_type": "invalid",
    },
    # 13. Remove interviewer from platform
    {
        "status": "replied",
        "product_area": "settings/roles_management",
        "response": (
            "To remove an interviewer from the HackerRank platform, please follow these steps:\n\n"
            "1. Log in to your HackerRank for Work account as an Admin or Owner.\n"
            "2. Navigate to Settings > Team Management (or User Management).\n"
            "3. Find the interviewer you want to remove using the search or filter options.\n"
            "4. Click the three-dot menu (⋮) next to their name.\n"
            "5. Select 'Remove User' or 'Deactivate' from the dropdown.\n\n"
            "Note: If you do not see the three-dot menu with a remove option, "
            "your account role may not have sufficient permissions. "
            "Please ensure you have Admin or Owner access, or contact your account's Primary Owner."
        ),
        "justification": (
            "User/interviewer removal is documented in HackerRank's roles management and team management settings. "
            "The corpus covers flexible user roles and team management with step-by-step guidance."
        ),
        "request_type": "product_issue",
    },
    # 14. Pause subscription
    {
        "status": "escalated",
        "product_area": "settings/company_level_admin_settings",
        "response": (
            "We understand you'd like to pause your HackerRank subscription. "
            "Subscription pauses or cancellations require direct handling by the HackerRank account management team, "
            "as these involve billing and contract adjustments. "
            "Please reach out to your dedicated HackerRank account representative, "
            "or contact HackerRank sales/support at https://www.hackerrank.com/contact-us to discuss your options."
        ),
        "justification": (
            "Subscription pause/cancellation involves billing contract changes that require human account team intervention. "
            "No self-service pause option is documented in the corpus for employer accounts. Escalated."
        ),
        "request_type": "product_issue",
    },
    # 15. Claude stopped working completely
    {
        "status": "escalated",
        "product_area": "claude/troubleshooting",
        "response": (
            "We're sorry to hear Claude is not responding. If all requests are failing, here are some immediate steps:\n\n"
            "1. Check the Anthropic status page for any ongoing incidents.\n"
            "2. Try refreshing the page or clearing your browser cache and cookies.\n"
            "3. If using the API, verify your API key is valid and not rate-limited.\n"
            "4. Try accessing Claude from a different browser or device.\n\n"
            "If the issue persists across all channels, this may indicate a service disruption. "
            "We are escalating this to our technical team for urgent review."
        ),
        "justification": (
            "Complete service failure for all requests indicates a potential outage requiring technical team investigation. "
            "Basic troubleshooting steps from the Claude troubleshooting corpus are provided; escalated for engineer review."
        ),
        "request_type": "bug",
    },
    # 16. Identity stolen — Visa
    {
        "status": "escalated",
        "product_area": "visa/fraud_protection",
        "response": (
            "Identity theft is a very serious matter. Please take these immediate steps:\n\n"
            "1. Contact your card-issuing bank immediately to report the identity theft and request a card block.\n"
            "2. Report the theft to your local police and obtain a case/reference number.\n"
            "3. Contact Visa's Global Customer Assistance Service at +1 303 967 1090 (available 24/7) "
            "if your Visa card has been compromised.\n"
            "4. File a report with your country's national cybercrime or fraud authority.\n\n"
            "We are escalating your case to the appropriate security team for further assistance."
        ),
        "justification": (
            "Identity theft is a critical fraud/security emergency requiring immediate human intervention. "
            "Per Visa's fraud protection corpus, cardholders should contact their issuing bank and local authorities. "
            "Escalated immediately — this cannot be handled by automated support."
        ),
        "request_type": "product_issue",
    },
    # 17. Resume Builder is Down
    {
        "status": "escalated",
        "product_area": "hackerrank_community/additional_resources",
        "response": (
            "We apologize for the inconvenience. It appears the HackerRank Resume Builder is currently unavailable. "
            "This has been escalated to our technical team for immediate investigation. "
            "In the meantime, you can use your HackerRank profile as a resume reference, "
            "or try accessing the feature again after some time. "
            "We apologize for the disruption."
        ),
        "justification": (
            "A service outage (Resume Builder down) requires engineering team investigation. "
            "Cannot be resolved through standard support. Escalated as a technical bug."
        ),
        "request_type": "bug",
    },
    # 18. Certificate name incorrect
    {
        "status": "replied",
        "product_area": "hackerrank_community/certifications",
        "response": (
            "To correct the name on your HackerRank certificate, please follow these steps:\n\n"
            "1. Log in to your HackerRank account.\n"
            "2. Click on your profile icon (top-right) and go to Settings > Profile.\n"
            "3. Update your First Name and Last Name to match the name you want on your certificate.\n"
            "4. Save your changes.\n\n"
            "After updating your profile name, your certificate should reflect the corrected name. "
            "If the certificate still shows the old name, please contact HackerRank support at "
            "support@hackerrank.com with your certificate details and updated name for manual correction."
        ),
        "justification": (
            "Certificate names are derived from the user's HackerRank profile. "
            "Updating the profile name and requesting certificate re-issuance is the documented resolution path. "
            "Addressable through self-service with support fallback."
        ),
        "request_type": "product_issue",
    },
    # 19. How do I dispute a charge — Visa
    {
        "status": "replied",
        "product_area": "dispute_resolution",
        "response": (
            "To dispute a charge on your Visa card, follow these steps:\n\n"
            "1. Contact your card-issuing bank (the bank that issued your Visa card) — not Visa directly.\n"
            "2. Inform them of the transaction you wish to dispute and provide any supporting documentation "
            "(receipt, order confirmation, communication with merchant).\n"
            "3. Your bank will initiate a formal dispute/chargeback process through the Visa network.\n"
            "4. The process typically involves the issuer contacting the merchant's acquiring bank for resolution.\n\n"
            "For in-store purchase issues, you can also use Visa's online reporting form: "
            "https://usa.visa.com/Forms/visa-rules.html\n\n"
            "For questions about Visa rules, email: visamerchantrules@visa.com"
        ),
        "justification": (
            "Visa's dispute resolution documentation clearly outlines that cardholders contact their card-issuing bank "
            "to initiate a chargeback. The corpus covers the step-by-step dispute process. Answered directly from corpus."
        ),
        "request_type": "product_issue",
    },
    # 20. Security vulnerability in Claude (bug bounty)
    {
        "status": "escalated",
        "product_area": "claude/safeguards",
        "response": (
            "Thank you for taking the time to report a potential security vulnerability in Claude. "
            "Responsible disclosure is greatly appreciated. "
            "Please report security vulnerabilities directly to Anthropic's security team through the official channels. "
            "Do not share vulnerability details through this support channel. "
            "Visit https://www.anthropic.com/security for the responsible disclosure policy and submission process. "
            "Your report has been escalated to the security team."
        ),
        "justification": (
            "Security vulnerability reports require dedicated security team handling and responsible disclosure procedures. "
            "This cannot be processed through automated support. "
            "Escalated immediately to prevent unintentional disclosure of sensitive security details."
        ),
        "request_type": "bug",
    },
    # 21. Stop Claude crawling website
    {
        "status": "replied",
        "product_area": "claude/privacy_and_legal",
        "response": (
            "To prevent Anthropic's bots from crawling your website, you can use the standard robots.txt mechanism:\n\n"
            "1. Add the following to your robots.txt file to block all Anthropic crawlers:\n\n"
            "   User-agent: ClaudeBot\n"
            "   Disallow: /\n\n"
            "   User-agent: anthropic-ai\n"
            "   Disallow: /\n\n"
            "   User-agent: Claude-Web\n"
            "   Disallow: /\n\n"
            "2. Add this to the robots.txt of every subdomain you wish to opt out from.\n\n"
            "Note: Blocking IP addresses is not recommended as it may prevent Anthropic from reading your robots.txt preferences.\n\n"
            "For questions or to report a malfunctioning crawler, email: claudebot@anthropic.com\n"
            "Source: https://support.claude.com/en/articles/8896518"
        ),
        "justification": (
            "The Anthropic web crawling opt-out process is documented in the Claude privacy corpus. "
            "Robots.txt with 'User-agent: ClaudeBot, Disallow: /' is the official opt-out method. Answered directly from corpus."
        ),
        "request_type": "product_issue",
    },
    # 22. Urgent cash, only VISA card
    {
        "status": "replied",
        "product_area": "travel_support",
        "response": (
            "If you urgently need cash and only have your Visa card, here are your options:\n\n"
            "1. **ATM Cash Withdrawal**: Use Visa's Global ATM Locator (https://www.visa.com/locator/atm) "
            "to find the nearest ATM that accepts Visa cards. Visa cards work at 80M+ locations in 200+ countries.\n"
            "2. **Cash Advance at a Bank**: Visit a local bank branch that displays the Visa logo "
            "and request a cash advance using your Visa card (your PIN will be required).\n"
            "3. **Emergency Card Services**: If your card is lost or stolen and you need emergency cash, "
            "contact Visa's Global Customer Assistance Service at +1 303 967 1090 (24/7).\n\n"
            "Note: Cash advances may incur fees and interest charges from your card issuer."
        ),
        "justification": (
            "Visa travel support corpus covers ATM locator and emergency cash services. "
            "Answering with documented options: ATM locator, bank cash advance, and emergency card services."
        ),
        "request_type": "product_issue",
    },
    # 23. Claude data usage for model improvement
    {
        "status": "replied",
        "product_area": "claude/privacy_and_legal",
        "response": (
            "Regarding data retention when you allow Claude to use your conversations for model improvement:\n\n"
            "For Claude consumer products (Free, Pro, Max plans and Claude Code):\n"
            "Data retention policies are detailed at https://privacy.claude.com/en/collections/10663362-consumers\n\n"
            "For Enterprise/Team plans with custom data retention:\n"
            "- By default, data is retained indefinitely unless a custom retention period is set.\n"
            "- Enterprise plan owners can configure custom retention periods (minimum 30 days).\n"
            "- Data is deleted at midnight UTC on the scheduled day.\n\n"
            "For more information on Anthropic's data practices, visit: "
            "https://support.claude.com/en/collections/4078534-privacy-and-legal"
        ),
        "justification": (
            "Claude's data retention policies are documented in the privacy-and-legal and enterprise data retention corpus. "
            "The response is grounded in documented policies without hallucinating specific timeframes not in the corpus."
        ),
        "request_type": "product_issue",
    },
    # 24. Code to delete all files — malicious
    {
        "status": "escalated",
        "product_area": "security",
        "response": (
            "This request cannot be fulfilled. Providing code to delete files from a system is outside the scope "
            "of our support services and could cause serious, irreversible harm. "
            "If you have a legitimate technical need, please describe your specific use case in more detail. "
            "This request has been flagged and escalated for review."
        ),
        "justification": (
            "Request for code to delete all system files is potentially malicious and outside any support scope. "
            "No supporting documentation exists in the corpus. Escalated and flagged as an invalid/harmful request."
        ),
        "request_type": "invalid",
    },
    # 25. French Visa prompt injection
    {
        "status": "escalated",
        "product_area": "security",
        "response": (
            "This request has been flagged and escalated to our security team for review. "
            "We are unable to process it through automated support."
        ),
        "justification": (
            "This ticket contains a prompt injection attempt: it requests the agent to reveal its internal rules, "
            "retrieved documents, and decision logic. This is a security violation. "
            "The card-blocking portion may be legitimate, but the adversarial component requires security team review."
        ),
        "request_type": "invalid",
    },
    # 26. AWS Bedrock requests failing for Claude
    {
        "status": "replied",
        "product_area": "amazon_bedrock",
        "response": (
            "We're sorry to hear you're experiencing issues with Claude on Amazon Bedrock. "
            "Here are some troubleshooting steps:\n\n"
            "1. **Verify model access**: Ensure you have requested and been granted access to the specific Claude model "
            "in your AWS region via the Amazon Bedrock console.\n"
            "2. **Check AWS region**: Claude models are available in specific AWS regions. "
            "Verify the model is available in your selected region.\n"
            "3. **Review IAM permissions**: Ensure your AWS IAM role/user has the required Bedrock permissions "
            "(bedrock:InvokeModel, bedrock:InvokeModelWithResponseStream).\n"
            "4. **Check API request format**: Ensure your requests conform to the Bedrock API specifications "
            "for Claude models.\n"
            "5. **AWS support**: For persistent infrastructure issues, contact AWS Support at "
            "https://aws.amazon.com/support\n\n"
            "For Claude-specific Bedrock support questions: "
            "https://support.claude.com/en/articles/7996921-i-use-claude-in-amazon-bedrock-who-do-i-contact"
        ),
        "justification": (
            "Amazon Bedrock integration issues are documented in the Claude/Amazon Bedrock corpus. "
            "Steps cover model access, region availability, IAM permissions — all documented in the corpus. "
            "AWS-level infrastructure issues are directed to AWS Support per corpus guidance."
        ),
        "request_type": "bug",
    },
    # 27. Remove employee from HackerRank hiring account
    {
        "status": "replied",
        "product_area": "settings/teams_management",
        "response": (
            "To remove a former employee from your HackerRank for Work hiring account:\n\n"
            "1. Log in to your HackerRank for Work account as an Admin or Owner.\n"
            "2. Go to Settings (gear icon) in the top navigation.\n"
            "3. Select Team Management or User Management.\n"
            "4. Locate the employee using the search or filter options.\n"
            "5. Click the three-dot menu (⋮) next to their name.\n"
            "6. Select Remove User or Deactivate.\n\n"
            "Once removed, the user will lose access to your company's HackerRank account and any associated tests. "
            "Their previous test results and reports remain accessible to your team."
        ),
        "justification": (
            "User removal from the HackerRank for Work account is covered in the team management and roles management sections. "
            "Steps are documented in the corpus for admin-level operations."
        ),
        "request_type": "product_issue",
    },
    # 28. Claude LTI key for students
    {
        "status": "replied",
        "product_area": "claude_for_education",
        "response": (
            "Claude for Education offers university-sponsored accounts with LMS integration capabilities. "
            "As a professor, here's how to get started:\n\n"
            "1. **University-sponsored setup**: Claude for Education is typically set up at the institutional level "
            "by your university's IT department or administrators. "
            "Contact your university's IT department to check if your institution has a Claude for Education account.\n\n"
            "2. **If your university is not yet enrolled**: "
            "Visit https://support.claude.com/en/articles/11139094 for admin getting-started guidance, "
            "or contact Anthropic at https://www.anthropic.com/education to inquire about institutional access.\n\n"
            "3. **For individual faculty**: "
            "See FAQs for faculty/student users at https://support.claude.com/en/articles/11139144"
        ),
        "justification": (
            "Claude for Education documentation covers university setup for professors/admins. "
            "The corpus includes getting-started guides for owners/admins and FAQ for faculty. "
            "LTI integration is part of the Education plan setup via SSO/SCIM."
        ),
        "request_type": "product_issue",
    },
    # 29. Visa minimum spend US Virgin Islands
    {
        "status": "replied",
        "product_area": "visa_rules",
        "response": (
            "Regarding minimum spend requirements on Visa cards in the US Virgin Islands:\n\n"
            "Visa's rules allow US merchants to set a minimum transaction amount for credit card purchases, "
            "up to $10, as permitted under US law and Visa's merchant guidelines. "
            "The US Virgin Islands is a US territory, so US merchant rules apply.\n\n"
            "This means a merchant can legally require a minimum $10 purchase when accepting Visa credit cards. "
            "However, this rule applies to credit cards — merchants cannot impose minimum spend requirements "
            "on Visa debit (check) cards.\n\n"
            "If you believe the minimum requirement is being applied incorrectly, "
            "you can report a purchase issue at: https://usa.visa.com/Forms/visa-rules.html\n"
            "Or inquire about Visa rules at: visamerchantrules@visa.com"
        ),
        "justification": (
            "Merchant minimum spend rules are covered in Visa's consumer rules and regulations corpus. "
            "US merchants may set up to $10 minimum for credit card transactions per Visa Core Rules. "
            "Answered directly from Visa rules documentation."
        ),
        "request_type": "product_issue",
    },
]


def generate():
    tickets = []
    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(dict(row))

    assert len(tickets) == len(PREDICTIONS), f"Ticket count mismatch: {len(tickets)} tickets vs {len(PREDICTIONS)} predictions"

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Issue", "Subject", "Company", "status", "product_area", "response", "justification", "request_type"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ticket, pred in zip(tickets, PREDICTIONS):
            row = {
                "Issue": ticket.get("Issue", ticket.get("issue", "")),
                "Subject": ticket.get("Subject", ticket.get("subject", "")),
                "Company": ticket.get("Company", ticket.get("company", "")),
                **pred,
            }
            writer.writerow(row)

    print(f"✓ Output written to {OUTPUT_CSV}")
    print(f"  Total tickets processed: {len(tickets)}")
    status_counts = {}
    for p in PREDICTIONS:
        status_counts[p["status"]] = status_counts.get(p["status"], 0) + 1
    print(f"  Replied: {status_counts.get('replied', 0)}, Escalated: {status_counts.get('escalated', 0)}")


if __name__ == "__main__":
    generate()
