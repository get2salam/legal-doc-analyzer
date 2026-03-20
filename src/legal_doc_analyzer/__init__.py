"""Legal Document Analyzer -- AI-powered legal document analysis."""

__version__ = "0.6.0"

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
from .segmenter import (
    DocumentSegmenter,
    DocumentStructure,
    Section,
    SectionLevel,
    segment_document,
)
from .summarizer import LegalSummarizer, SentenceScore, SummaryResult
from .timeline import (
    ContractTimeline,
    EventType,
    TimelineEvent,
    TimelineExtractor,
)

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
    # Document segmentation
    "DocumentSegmenter",
    "DocumentStructure",
    "Section",
    "SectionLevel",
    "segment_document",
    # Summarization
    "LegalSummarizer",
    "SentenceScore",
    "SummaryResult",
    # Timeline extraction
    "TimelineExtractor",
    "ContractTimeline",
    "TimelineEvent",
    "EventType",
]
