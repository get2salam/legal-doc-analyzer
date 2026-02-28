"""
LLM-based Legal Document Analyzer
Uses OpenAI or Anthropic models to analyze legal documents.
"""

import json
import os

from tenacity import retry, stop_after_attempt, wait_exponential

from prompts.analysis_prompts import (
    ANALYSIS_PROMPT,
    DEEP_ANALYSIS_PROMPT,
    QUICK_ANALYSIS_PROMPT,
    SYSTEM_PROMPT,
)
from utils.document_processor import chunk_text, estimate_tokens


class LegalAnalyzer:
    """
    Analyzes legal documents using LLM models.
    Supports both OpenAI and Anthropic models.
    """

    def __init__(self, model: str = "gpt-4-turbo"):
        """
        Initialize the analyzer with the specified model.

        Args:
            model: Model name (gpt-4, gpt-3.5-turbo, claude-3-sonnet, etc.)
        """
        self.model = model
        self.is_anthropic = model.startswith("claude")
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate API client."""
        if self.is_anthropic:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            from anthropic import Anthropic

            self.client = Anthropic(api_key=api_key)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """
        Make a call to the LLM with retry logic.

        Args:
            prompt: The user prompt
            system_prompt: The system prompt

        Returns:
            The model's response text
        """
        if self.is_anthropic:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=4096,
            )
            return response.choices[0].message.content

    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from the LLM response, handling markdown code blocks.
        """
        # Try to extract JSON from markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a basic structure
            return {
                "contract_type": "Unknown",
                "overall_risk": "Unknown",
                "parties": [],
                "key_dates": {},
                "financial_terms": [],
                "jurisdiction": "Unable to parse",
                "key_clauses": [],
                "risk_flags": [],
                "plain_summary": response[:500],  # Use first 500 chars as summary
            }

    def analyze(self, document_text: str, depth: str = "Standard") -> dict:
        """
        Analyze a legal document.

        Args:
            document_text: The full text of the document
            depth: Analysis depth - "Quick", "Standard", or "Deep"

        Returns:
            Dictionary containing the analysis results
        """
        # Select prompt based on depth
        if depth == "Quick":
            prompt_template = QUICK_ANALYSIS_PROMPT
        elif depth == "Deep":
            prompt_template = DEEP_ANALYSIS_PROMPT
        else:
            prompt_template = ANALYSIS_PROMPT

        # Check if document is too large and needs chunking
        token_estimate = estimate_tokens(document_text)

        if token_estimate > 100000:  # Very large document
            return self._analyze_large_document(document_text, prompt_template)

        # Standard analysis
        prompt = prompt_template.format(document_text=document_text[:50000])  # Cap at 50k chars
        response = self._call_llm(prompt)

        return self._parse_json_response(response)

    def _analyze_large_document(self, document_text: str, prompt_template: str) -> dict:
        """
        Analyze a large document by chunking and combining results.
        """
        chunks = chunk_text(document_text, chunk_size=20000, overlap=500)

        # Analyze each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"""This is part {i + 1} of {len(chunks)} of a legal document.
            
{prompt_template.format(document_text=chunk)}

Note: This is a partial document. Focus on what's present in this section."""

            response = self._call_llm(chunk_prompt)
            chunk_results.append(self._parse_json_response(response))

        # Combine results
        return self._merge_chunk_results(chunk_results)

    def _merge_chunk_results(self, results: list[dict]) -> dict:
        """Merge analysis results from multiple chunks."""
        merged = {
            "contract_type": results[0].get("contract_type", "Unknown"),
            "overall_risk": "Medium",  # Will be recalculated
            "parties": [],
            "key_dates": {},
            "financial_terms": [],
            "jurisdiction": "",
            "key_clauses": [],
            "risk_flags": [],
            "plain_summary": "",
        }

        seen_parties = set()
        summaries = []
        risk_levels = []

        for result in results:
            # Merge parties (deduplicate by name)
            for party in result.get("parties", []):
                if party.get("name") not in seen_parties:
                    seen_parties.add(party.get("name"))
                    merged["parties"].append(party)

            # Merge dates
            merged["key_dates"].update(result.get("key_dates", {}))

            # Merge financial terms
            merged["financial_terms"].extend(result.get("financial_terms", []))

            # Keep jurisdiction if found
            if result.get("jurisdiction") and not merged["jurisdiction"]:
                merged["jurisdiction"] = result["jurisdiction"]

            # Merge clauses and risks
            merged["key_clauses"].extend(result.get("key_clauses", []))
            merged["risk_flags"].extend(result.get("risk_flags", []))

            # Collect summaries and risk levels
            if result.get("plain_summary"):
                summaries.append(result["plain_summary"])
            if result.get("overall_risk"):
                risk_levels.append(result["overall_risk"])

        # Calculate overall risk
        if risk_levels:
            if "High" in risk_levels:
                merged["overall_risk"] = "High"
            elif risk_levels.count("Medium") > len(risk_levels) // 2:
                merged["overall_risk"] = "Medium"
            else:
                merged["overall_risk"] = "Low"

        # Combine summaries
        merged["plain_summary"] = " ".join(summaries)[:1000]

        return merged
