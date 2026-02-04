"""Legal Document Analyzer — AI-powered legal document analysis."""

__version__ = "0.1.0"

from .analyzer import LegalAnalyzer
from .models import AnalysisResult, Clause, Entity, Risk

__all__ = ["LegalAnalyzer", "AnalysisResult", "Clause", "Entity", "Risk"]
