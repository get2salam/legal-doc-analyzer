"""
Prompt templates for legal document analysis.
"""

SYSTEM_PROMPT = """You are an expert legal analyst with extensive experience reviewing contracts, 
agreements, and legal documents. You have deep knowledge of contract law, commercial transactions, 
and risk assessment. Your analysis is thorough, practical, and accessible to non-lawyers.

Key principles:
- Be accurate and cite specific sections when relevant
- Flag genuine risks without being alarmist
- Explain legal terms in plain English
- Consider both parties' perspectives
- Highlight unusual or non-standard terms
"""

ANALYSIS_PROMPT = """Analyze the following legal document and provide a comprehensive assessment.

DOCUMENT:
{document_text}

Provide your analysis in the following JSON format (respond ONLY with valid JSON):

{{
    "contract_type": "Type of agreement (e.g., NDA, Employment, Service Agreement)",
    "overall_risk": "Low/Medium/High",
    "parties": [
        {{"name": "Party name", "role": "Role in agreement (e.g., Provider, Client)"}}
    ],
    "key_dates": {{
        "Effective Date": "date or description",
        "Termination Date": "date or description",
        "Other important dates": "as relevant"
    }},
    "financial_terms": [
        {{"type": "Type of payment", "amount": "Amount", "details": "Additional details"}}
    ],
    "jurisdiction": "Governing law and dispute resolution venue",
    "key_clauses": [
        {{
            "title": "Clause name",
            "summary": "What this clause means in plain English",
            "risk": "Low/Medium/High",
            "recommendation": "Specific advice for this clause"
        }}
    ],
    "risk_flags": [
        {{
            "severity": "high/medium/low",
            "issue": "Brief issue title",
            "description": "Detailed explanation of the concern"
        }}
    ],
    "plain_summary": "A 2-3 paragraph summary of the document in plain English, highlighting the most important points a non-lawyer should understand"
}}

Focus on:
1. Identifying all parties and their obligations
2. Finding any unusual or one-sided terms
3. Highlighting potential risks and liabilities
4. Noting any missing standard protections
5. Explaining complex terms in accessible language
"""

QUICK_ANALYSIS_PROMPT = """Quickly analyze this legal document and provide key highlights.

DOCUMENT:
{document_text}

Provide a concise analysis in JSON format:

{{
    "contract_type": "Type of agreement",
    "overall_risk": "Low/Medium/High",
    "parties": [
        {{"name": "Party name", "role": "Role"}}
    ],
    "key_dates": {{}},
    "financial_terms": [],
    "jurisdiction": "Governing law",
    "key_clauses": [
        {{
            "title": "Most important clause",
            "summary": "Brief summary",
            "risk": "Risk level",
            "recommendation": "Key advice"
        }}
    ],
    "risk_flags": [
        {{
            "severity": "level",
            "issue": "Top concern",
            "description": "Brief description"
        }}
    ],
    "plain_summary": "One paragraph summary of key points"
}}

Focus on the 3-5 most important aspects only.
"""

DEEP_ANALYSIS_PROMPT = """Perform an exhaustive legal analysis of this document.

DOCUMENT:
{document_text}

Provide a comprehensive analysis in JSON format:

{{
    "contract_type": "Specific type and subtype of agreement",
    "overall_risk": "Low/Medium/High with confidence level",
    "parties": [
        {{
            "name": "Full legal name",
            "role": "Detailed role and obligations",
            "address": "If mentioned",
            "contact": "If mentioned"
        }}
    ],
    "key_dates": {{
        "Effective Date": "Exact date and any conditions",
        "Termination Date": "Exact date or term",
        "Notice Periods": "All notice requirements",
        "Renewal Terms": "Auto-renewal or manual",
        "Other Dates": "Any other time-sensitive provisions"
    }},
    "financial_terms": [
        {{
            "type": "Category of financial term",
            "amount": "Exact amount or formula",
            "currency": "Currency if specified",
            "timing": "When due",
            "conditions": "Any conditions or adjustments",
            "details": "Full context"
        }}
    ],
    "jurisdiction": "Complete governing law clause including dispute resolution, arbitration provisions, and venue",
    "key_clauses": [
        {{
            "title": "Clause title",
            "section": "Section number if available",
            "summary": "Detailed plain English explanation",
            "legal_implications": "What this means legally",
            "risk": "Low/Medium/High",
            "market_standard": "Is this typical?",
            "recommendation": "Specific negotiation advice",
            "suggested_revision": "Alternative language if concerning"
        }}
    ],
    "risk_flags": [
        {{
            "severity": "high/medium/low",
            "issue": "Issue title",
            "description": "Detailed explanation",
            "legal_basis": "Why this is a concern",
            "mitigation": "How to address this"
        }}
    ],
    "missing_provisions": [
        "List of standard clauses that are absent but might be expected"
    ],
    "ambiguities": [
        "List of unclear terms or provisions that could lead to disputes"
    ],
    "compliance_considerations": [
        "Any regulatory or compliance issues (GDPR, employment law, etc.)"
    ],
    "plain_summary": "Comprehensive 3-4 paragraph summary covering all key aspects, risks, and recommendations in accessible language"
}}

Be thorough and analytical. Consider:
1. Every party's rights and obligations
2. All financial implications
3. Termination scenarios and consequences
4. Liability allocation and limitations
5. IP rights and confidentiality
6. Compliance with applicable laws
7. Comparison to market standards
8. Negotiation leverage points
"""
