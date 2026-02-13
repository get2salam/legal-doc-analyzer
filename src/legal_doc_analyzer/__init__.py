"""Legal Document Analyzer — AI-powered legal document analysis."""

__version__ = "0.2.0"

from .analyzer import LegalAnalyzer
from .models import AnalysisResult, Clause, Entity, Risk
from .preprocessing import (
    ComparisonResult,
    ReadabilityResult,
    TextPreprocessor,
    compare_documents,
    count_syllables,
)

__all__ = [
    "LegalAnalyzer",
    "AnalysisResult",
    "Clause",
    "Entity",
    "Risk",
    "TextPreprocessor",
    "ReadabilityResult",
    "ComparisonResult",
    "compare_documents",
    "count_syllables",
]
