"""Legal Document Analyzer -- AI-powered legal document analysis."""

__version__ = "0.5.0"

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
from .comparator import (
    ClauseDiff,
    ContractComparator,
    ContractDiff,
    EntityDelta,
)
from .models import AnalysisResult, Clause, Entity, Risk
from .preprocessing import (
    ComparisonResult,
    ReadabilityResult,
    TextPreprocessor,
    compare_documents,
    count_syllables,
)
from .summarizer import LegalSummarizer, SentenceScore, SummaryResult

__all__ = [
    # Core
    "LegalAnalyzer",
    "AnalysisResult",
    "Clause",
    "Entity",
    "Risk",
    # Contract comparison
    "ContractComparator",
    "ContractDiff",
    "ClauseDiff",
    "EntityDelta",
    # Preprocessing
    "TextPreprocessor",
    "ReadabilityResult",
    "ComparisonResult",
    "compare_documents",
    "count_syllables",
    # Classification
    "DocumentClassifier",
    "DocumentType",
    "TfidfVectorizer",
    "NaiveBayesClassifier",
    "ClassificationResult",
    "ClassificationMetrics",
    "compute_metrics",
    "cross_validate",
    "stratified_k_fold",
    # Summarization
    "LegalSummarizer",
    "SentenceScore",
    "SummaryResult",
]
