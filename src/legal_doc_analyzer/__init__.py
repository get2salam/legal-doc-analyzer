"""Legal Document Analyzer — AI-powered legal document analysis."""

__version__ = "0.3.0"

from .analyzer import LegalAnalyzer
from .classifier import (
    ClassificationMetrics,
    ClassificationResult,
    DocumentClassifier,
    DocumentType,
    NaiveBayesClassifier,
    TfidfVectorizer,
    compute_metrics,
    cross_validate,
    stratified_k_fold,
)
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
    "DocumentClassifier",
    "DocumentType",
    "TfidfVectorizer",
    "NaiveBayesClassifier",
    "ClassificationResult",
    "ClassificationMetrics",
    "compute_metrics",
    "cross_validate",
    "stratified_k_fold",
]
