"""Document classification pipeline with TF-IDF features and Naive Bayes.

Provides a complete ML pipeline for classifying legal documents by type
(contract, brief, statute, opinion, etc.) using pure Python — no sklearn
or numpy required.

Features:
- TF-IDF vectorization with configurable n-grams and vocabulary pruning
- Multinomial Naive Bayes with Laplace smoothing
- Stratified k-fold cross-validation
- Precision, recall, F1, and confusion matrix evaluation
- Model persistence (JSON serialization)
- Feature importance / most informative features per class

Designed as a lightweight baseline classifier suitable for bootstrap
labeling, quick prototyping, or production use on small-to-medium corpora.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Document Types
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    """Standard legal document categories."""

    CONTRACT = "contract"
    BRIEF = "brief"
    STATUTE = "statute"
    OPINION = "opinion"
    MEMORANDUM = "memorandum"
    PLEADING = "pleading"
    MOTION = "motion"
    ORDER = "order"
    REGULATION = "regulation"
    POLICY = "policy"
    LETTER = "letter"
    OTHER = "other"


# ---------------------------------------------------------------------------
# TF-IDF Vectorizer
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z'-]*[a-zA-Z]\b|\b[a-zA-Z]\b")

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "must",
    "not", "no", "nor", "so", "if", "then", "than", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "their",
    "his", "her", "our", "your", "we", "you", "who", "whom", "which",
    "what", "where", "when", "how", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "any", "only",
    "own", "same", "too", "very", "just", "about", "above", "after",
    "again", "also", "because", "before", "between", "during", "into",
    "through", "under", "until", "up", "out", "over", "here", "there",
})


def _tokenize(text: str) -> list[str]:
    """Extract lowercase word tokens from text."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    """Generate n-grams from a token list."""
    if n <= 1:
        return tokens
    return ["_".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


@dataclass
class TfidfVectorizer:
    """Pure-Python TF-IDF vectorizer with n-gram support.

    Builds a vocabulary from training documents and transforms text into
    TF-IDF weighted feature vectors (represented as sparse dicts).

    Args:
        max_features: Maximum vocabulary size (most frequent terms kept).
        min_df: Minimum document frequency (absolute count) for a term.
        max_df_ratio: Maximum document frequency as fraction of corpus size.
        ngram_range: Tuple of (min_n, max_n) for n-gram generation.
        use_stopwords: Whether to filter English stopwords.
        sublinear_tf: Use ``1 + log(tf)`` instead of raw term frequency.
    """

    max_features: int = 5000
    min_df: int = 2
    max_df_ratio: float = 0.95
    ngram_range: tuple[int, int] = (1, 2)
    use_stopwords: bool = True
    sublinear_tf: bool = True

    # Learned state
    vocabulary_: dict[str, int] = field(default_factory=dict, repr=False)
    idf_: dict[str, float] = field(default_factory=dict, repr=False)
    _num_docs: int = 0

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        """Learn vocabulary and IDF weights from a corpus.

        Args:
            documents: List of raw text documents.

        Returns:
            Self (for method chaining).
        """
        n_docs = len(documents)
        self._num_docs = n_docs

        # Count document frequency for each term
        doc_freq: Counter[str] = Counter()
        for doc in documents:
            terms = self._extract_terms(doc)
            unique_terms = set(terms)
            for term in unique_terms:
                doc_freq[term] += 1

        # Prune by min/max document frequency
        max_df_abs = max(1, int(self.max_df_ratio * n_docs))
        pruned = {
            term: df
            for term, df in doc_freq.items()
            if df >= self.min_df and df <= max_df_abs
        }

        # Select top features by frequency
        sorted_terms = sorted(pruned.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[: self.max_features]

        # Build vocabulary mapping
        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(top_terms)}

        # Compute IDF: log((1 + N) / (1 + df)) + 1 (smooth IDF)
        self.idf_ = {}
        for term, idx in self.vocabulary_.items():
            df = doc_freq.get(term, 0)
            self.idf_[term] = math.log((1 + n_docs) / (1 + df)) + 1

        return self

    def transform(self, documents: list[str]) -> list[dict[str, float]]:
        """Transform documents into TF-IDF feature vectors.

        Args:
            documents: List of raw text documents.

        Returns:
            List of sparse feature vectors (dicts of {term: tfidf_weight}).

        Raises:
            RuntimeError: If the vectorizer has not been fitted.
        """
        if not self.vocabulary_:
            raise RuntimeError("Vectorizer has not been fitted. Call fit() first.")

        vectors = []
        for doc in documents:
            terms = self._extract_terms(doc)
            tf = Counter(terms)

            vec: dict[str, float] = {}
            for term in tf:
                if term not in self.vocabulary_:
                    continue
                raw_tf = tf[term]
                if self.sublinear_tf:
                    weighted_tf = 1 + math.log(raw_tf) if raw_tf > 0 else 0
                else:
                    weighted_tf = raw_tf
                vec[term] = weighted_tf * self.idf_.get(term, 0)

            # L2 normalization
            norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
            vec = {k: v / norm for k, v in vec.items()}

            vectors.append(vec)

        return vectors

    def fit_transform(self, documents: list[str]) -> list[dict[str, float]]:
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)

    def _extract_terms(self, text: str) -> list[str]:
        """Tokenize and generate n-grams from text."""
        tokens = _tokenize(text)
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in _STOP_WORDS]

        all_terms: list[str] = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            all_terms.extend(_ngrams(tokens, n))

        return all_terms

    def to_dict(self) -> dict:
        """Serialize vectorizer state to a dictionary."""
        return {
            "max_features": self.max_features,
            "min_df": self.min_df,
            "max_df_ratio": self.max_df_ratio,
            "ngram_range": list(self.ngram_range),
            "use_stopwords": self.use_stopwords,
            "sublinear_tf": self.sublinear_tf,
            "vocabulary": self.vocabulary_,
            "idf": self.idf_,
            "num_docs": self._num_docs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TfidfVectorizer":
        """Deserialize vectorizer from a dictionary."""
        vec = cls(
            max_features=data["max_features"],
            min_df=data["min_df"],
            max_df_ratio=data["max_df_ratio"],
            ngram_range=tuple(data["ngram_range"]),
            use_stopwords=data["use_stopwords"],
            sublinear_tf=data["sublinear_tf"],
        )
        vec.vocabulary_ = data["vocabulary"]
        vec.idf_ = data["idf"]
        vec._num_docs = data["num_docs"]
        return vec


# ---------------------------------------------------------------------------
# Multinomial Naive Bayes
# ---------------------------------------------------------------------------

@dataclass
class NaiveBayesClassifier:
    """Multinomial Naive Bayes classifier with Laplace smoothing.

    Works directly with sparse TF-IDF vectors (dicts). Suitable for
    text classification tasks where features are non-negative weights.

    Args:
        alpha: Laplace smoothing parameter (1.0 = standard smoothing).
    """

    alpha: float = 1.0

    # Learned parameters
    classes_: list[str] = field(default_factory=list, repr=False)
    class_log_prior_: dict[str, float] = field(default_factory=dict, repr=False)
    feature_log_prob_: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    _vocabulary: list[str] = field(default_factory=list, repr=False)

    def fit(
        self,
        vectors: list[dict[str, float]],
        labels: list[str],
    ) -> "NaiveBayesClassifier":
        """Train the classifier on TF-IDF vectors and labels.

        Args:
            vectors: List of sparse feature vectors from TfidfVectorizer.
            labels: List of class labels (same length as vectors).

        Returns:
            Self (for method chaining).

        Raises:
            ValueError: If vectors and labels have different lengths.
        """
        if len(vectors) != len(labels):
            raise ValueError(
                f"vectors ({len(vectors)}) and labels ({len(labels)}) must have same length"
            )

        # Collect all feature names
        all_features: set[str] = set()
        for vec in vectors:
            all_features.update(vec.keys())
        self._vocabulary = sorted(all_features)

        # Group vectors by class
        class_vectors: dict[str, list[dict[str, float]]] = defaultdict(list)
        for vec, label in zip(vectors, labels):
            class_vectors[label].append(vec)

        self.classes_ = sorted(class_vectors.keys())
        n_total = len(labels)

        # Compute log priors: log(P(class))
        self.class_log_prior_ = {}
        for cls in self.classes_:
            self.class_log_prior_[cls] = math.log(len(class_vectors[cls]) / n_total)

        # Compute feature log probabilities per class
        # P(feature|class) = (sum of feature weights in class + alpha) / (total weights in class + alpha * |V|)
        vocab_size = len(self._vocabulary)
        self.feature_log_prob_ = {}

        for cls in self.classes_:
            # Sum feature weights across all documents in this class
            feature_sums: dict[str, float] = defaultdict(float)
            for vec in class_vectors[cls]:
                for feat, weight in vec.items():
                    feature_sums[feat] += weight

            total_weight = sum(feature_sums.values())
            denominator = total_weight + self.alpha * vocab_size

            log_probs: dict[str, float] = {}
            for feat in self._vocabulary:
                numerator = feature_sums.get(feat, 0) + self.alpha
                log_probs[feat] = math.log(numerator / denominator)

            self.feature_log_prob_[cls] = log_probs

        return self

    def predict(self, vectors: list[dict[str, float]]) -> list[str]:
        """Predict class labels for a batch of feature vectors.

        Args:
            vectors: List of sparse feature vectors.

        Returns:
            List of predicted class labels.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        if not self.classes_:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")
        return [self._predict_single(vec) for vec in vectors]

    def predict_proba(self, vectors: list[dict[str, float]]) -> list[dict[str, float]]:
        """Predict class probabilities for a batch of feature vectors.

        Uses log-sum-exp for numerical stability.

        Args:
            vectors: List of sparse feature vectors.

        Returns:
            List of dicts mapping class names to probabilities.
        """
        if not self.classes_:
            raise RuntimeError("Classifier has not been fitted. Call fit() first.")
        return [self._predict_proba_single(vec) for vec in vectors]

    def _predict_single(self, vec: dict[str, float]) -> str:
        """Predict class for a single vector."""
        scores = self._compute_log_scores(vec)
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def _predict_proba_single(self, vec: dict[str, float]) -> dict[str, float]:
        """Compute class probabilities for a single vector."""
        log_scores = self._compute_log_scores(vec)

        # Log-sum-exp for numerical stability
        max_score = max(log_scores.values())
        exp_scores = {cls: math.exp(s - max_score) for cls, s in log_scores.items()}
        total = sum(exp_scores.values())

        return {cls: score / total for cls, score in exp_scores.items()}

    def _compute_log_scores(self, vec: dict[str, float]) -> dict[str, float]:
        """Compute unnormalized log posterior scores for each class."""
        scores: dict[str, float] = {}
        for cls in self.classes_:
            score = self.class_log_prior_[cls]
            log_probs = self.feature_log_prob_[cls]
            for feat, weight in vec.items():
                if feat in log_probs:
                    score += weight * log_probs[feat]
            scores[cls] = score
        return scores

    def most_informative_features(
        self,
        class_name: str,
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Return the most discriminative features for a given class.

        Measures how much more likely a feature is under the target class
        compared to the average of all other classes.

        Args:
            class_name: Target class name.
            top_n: Number of top features to return.

        Returns:
            List of (feature_name, log_likelihood_ratio) tuples, sorted
            by discriminative power (descending).

        Raises:
            ValueError: If class_name is not in the fitted classes.
        """
        if class_name not in self.classes_:
            raise ValueError(f"Unknown class: {class_name}. Known: {self.classes_}")

        target_probs = self.feature_log_prob_[class_name]
        other_classes = [c for c in self.classes_ if c != class_name]

        if not other_classes:
            return [(feat, prob) for feat, prob in sorted(
                target_probs.items(), key=lambda x: x[1], reverse=True
            )][:top_n]

        ratios: list[tuple[str, float]] = []
        for feat in self._vocabulary:
            target_lp = target_probs.get(feat, -20)
            # Average log prob across other classes
            other_lps = [
                self.feature_log_prob_[c].get(feat, -20) for c in other_classes
            ]
            avg_other_lp = sum(other_lps) / len(other_lps)
            ratio = target_lp - avg_other_lp
            ratios.append((feat, round(ratio, 4)))

        ratios.sort(key=lambda x: x[1], reverse=True)
        return ratios[:top_n]

    def to_dict(self) -> dict:
        """Serialize classifier state."""
        return {
            "alpha": self.alpha,
            "classes": self.classes_,
            "class_log_prior": self.class_log_prior_,
            "feature_log_prob": self.feature_log_prob_,
            "vocabulary": self._vocabulary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NaiveBayesClassifier":
        """Deserialize classifier from a dictionary."""
        nb = cls(alpha=data["alpha"])
        nb.classes_ = data["classes"]
        nb.class_log_prior_ = data["class_log_prior"]
        nb.feature_log_prob_ = data["feature_log_prob"]
        nb._vocabulary = data["vocabulary"]
        return nb


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

@dataclass
class ClassificationMetrics:
    """Evaluation metrics for a classification result.

    Attributes:
        accuracy: Overall accuracy.
        per_class: Per-class precision, recall, F1 scores.
        macro_precision: Unweighted mean precision across classes.
        macro_recall: Unweighted mean recall across classes.
        macro_f1: Unweighted mean F1 across classes.
        weighted_f1: Support-weighted mean F1 across classes.
        confusion_matrix: Dict of {(true, predicted): count}.
        support: Per-class sample counts in the true labels.
    """

    accuracy: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "per_class": {
                cls: {k: round(v, 4) for k, v in metrics.items()}
                for cls, metrics in self.per_class.items()
            },
            "confusion_matrix": self.confusion_matrix,
        }

    def summary(self) -> str:
        """Human-readable summary of metrics."""
        lines = [
            f"Accuracy: {self.accuracy:.2%}",
            f"Macro F1: {self.macro_f1:.4f}",
            f"Weighted F1: {self.weighted_f1:.4f}",
            "",
            f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
            "-" * 62,
        ]
        for cls in sorted(self.per_class.keys()):
            m = self.per_class[cls]
            s = self.support.get(cls, 0)
            lines.append(
                f"{cls:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {s:>10}"
            )
        return "\n".join(lines)


def compute_metrics(
    y_true: list[str],
    y_pred: list[str],
) -> ClassificationMetrics:
    """Compute classification metrics from true and predicted labels.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        ClassificationMetrics with accuracy, per-class, and aggregate scores.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    classes = sorted(set(y_true) | set(y_pred))
    n = len(y_true)

    # Confusion matrix
    cm: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    support: dict[str, int] = Counter(y_true)

    for cls in classes:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in classes if other != cls)
        fn = sum(cm[cls][other] for other in classes if other != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Macro averages
    macro_p = sum(m["precision"] for m in per_class.values()) / len(classes) if classes else 0
    macro_r = sum(m["recall"] for m in per_class.values()) / len(classes) if classes else 0
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(classes) if classes else 0

    # Weighted F1
    total_support = sum(support.values())
    weighted_f1 = (
        sum(per_class[cls]["f1"] * support.get(cls, 0) for cls in classes) / total_support
        if total_support > 0
        else 0
    )

    return ClassificationMetrics(
        accuracy=accuracy,
        per_class=per_class,
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        confusion_matrix=cm,
        support=dict(support),
    )


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------

def stratified_k_fold(
    labels: list[str],
    k: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Generate stratified k-fold train/test index splits.

    Ensures each fold has approximately the same class distribution
    as the full dataset.

    Args:
        labels: List of class labels.
        k: Number of folds.
        seed: Random seed for reproducibility.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    rng = random.Random(seed)

    # Group indices by class
    class_indices: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Shuffle within each class
    for indices in class_indices.values():
        rng.shuffle(indices)

    # Assign each index to a fold (round-robin within class)
    fold_assignments: list[int] = [0] * len(labels)
    for cls_indices in class_indices.values():
        for i, idx in enumerate(cls_indices):
            fold_assignments[idx] = i % k

    # Build fold splits
    folds: list[tuple[list[int], list[int]]] = []
    for fold_idx in range(k):
        test_indices = [i for i, f in enumerate(fold_assignments) if f == fold_idx]
        train_indices = [i for i, f in enumerate(fold_assignments) if f != fold_idx]
        folds.append((train_indices, test_indices))

    return folds


def cross_validate(
    documents: list[str],
    labels: list[str],
    k: int = 5,
    vectorizer_kwargs: Optional[dict] = None,
    classifier_kwargs: Optional[dict] = None,
    seed: int = 42,
) -> list[ClassificationMetrics]:
    """Run stratified k-fold cross-validation.

    Trains a TF-IDF + Naive Bayes pipeline on each fold and returns
    evaluation metrics for each test set.

    Args:
        documents: List of raw text documents.
        labels: List of corresponding class labels.
        k: Number of folds.
        vectorizer_kwargs: Optional kwargs for TfidfVectorizer.
        classifier_kwargs: Optional kwargs for NaiveBayesClassifier.
        seed: Random seed for fold generation.

    Returns:
        List of ClassificationMetrics (one per fold).
    """
    vec_kwargs = vectorizer_kwargs or {}
    cls_kwargs = classifier_kwargs or {}

    folds = stratified_k_fold(labels, k=k, seed=seed)
    results: list[ClassificationMetrics] = []

    for train_idx, test_idx in folds:
        train_docs = [documents[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_docs = [documents[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        # Build pipeline
        vectorizer = TfidfVectorizer(**vec_kwargs)
        train_vectors = vectorizer.fit_transform(train_docs)
        test_vectors = vectorizer.transform(test_docs)

        classifier = NaiveBayesClassifier(**cls_kwargs)
        classifier.fit(train_vectors, train_labels)
        predictions = classifier.predict(test_vectors)

        metrics = compute_metrics(test_labels, predictions)
        results.append(metrics)

    return results


# ---------------------------------------------------------------------------
# Classification Pipeline (High-Level API)
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """Result of classifying a single document."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "probabilities": {
                k: round(v, 4) for k, v in sorted(
                    self.probabilities.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            },
        }


class DocumentClassifier:
    """High-level document classification pipeline.

    Wraps TF-IDF vectorization and Naive Bayes classification into
    a simple train/predict interface with model persistence.

    Example::

        classifier = DocumentClassifier()
        classifier.train(documents, labels)

        result = classifier.classify("This agreement is entered into...")
        print(result.predicted_class)  # "contract"
        print(result.confidence)       # 0.87

        # Save model
        classifier.save("model.json")

        # Load model
        loaded = DocumentClassifier.load("model.json")

    Args:
        vectorizer_kwargs: Optional configuration for TfidfVectorizer.
        classifier_kwargs: Optional configuration for NaiveBayesClassifier.
    """

    def __init__(
        self,
        vectorizer_kwargs: Optional[dict] = None,
        classifier_kwargs: Optional[dict] = None,
    ) -> None:
        self._vec_kwargs = vectorizer_kwargs or {}
        self._cls_kwargs = classifier_kwargs or {}
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._classifier: Optional[NaiveBayesClassifier] = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Whether the classifier has been trained."""
        return self._is_trained

    @property
    def classes(self) -> list[str]:
        """List of known classes."""
        if self._classifier:
            return self._classifier.classes_
        return []

    def train(
        self,
        documents: list[str],
        labels: list[str],
    ) -> ClassificationMetrics:
        """Train the classifier and return training metrics.

        Args:
            documents: List of raw text documents.
            labels: Corresponding class labels.

        Returns:
            ClassificationMetrics on the training set (for sanity checking).
        """
        self._vectorizer = TfidfVectorizer(**self._vec_kwargs)
        vectors = self._vectorizer.fit_transform(documents)

        self._classifier = NaiveBayesClassifier(**self._cls_kwargs)
        self._classifier.fit(vectors, labels)
        self._is_trained = True

        # Return training accuracy as sanity check
        predictions = self._classifier.predict(vectors)
        return compute_metrics(labels, predictions)

    def classify(self, text: str) -> ClassificationResult:
        """Classify a single document.

        Args:
            text: Raw document text.

        Returns:
            ClassificationResult with predicted class and probabilities.

        Raises:
            RuntimeError: If the classifier has not been trained.
        """
        if not self._is_trained or not self._vectorizer or not self._classifier:
            raise RuntimeError("Classifier not trained. Call train() first.")

        vectors = self._vectorizer.transform([text])
        proba = self._classifier.predict_proba(vectors)[0]
        predicted = max(proba, key=proba.get)  # type: ignore[arg-type]
        confidence = proba[predicted]

        return ClassificationResult(
            predicted_class=predicted,
            confidence=confidence,
            probabilities=proba,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify multiple documents.

        Args:
            texts: List of raw document texts.

        Returns:
            List of ClassificationResult objects.
        """
        if not self._is_trained or not self._vectorizer or not self._classifier:
            raise RuntimeError("Classifier not trained. Call train() first.")

        vectors = self._vectorizer.transform(texts)
        probas = self._classifier.predict_proba(vectors)

        results = []
        for proba in probas:
            predicted = max(proba, key=proba.get)  # type: ignore[arg-type]
            results.append(ClassificationResult(
                predicted_class=predicted,
                confidence=proba[predicted],
                probabilities=proba,
            ))
        return results

    def evaluate(
        self,
        documents: list[str],
        labels: list[str],
        k: int = 5,
        seed: int = 42,
    ) -> list[ClassificationMetrics]:
        """Run k-fold cross-validation evaluation.

        Args:
            documents: List of raw text documents.
            labels: Corresponding class labels.
            k: Number of folds.
            seed: Random seed.

        Returns:
            List of ClassificationMetrics (one per fold).
        """
        return cross_validate(
            documents,
            labels,
            k=k,
            vectorizer_kwargs=self._vec_kwargs,
            classifier_kwargs=self._cls_kwargs,
            seed=seed,
        )

    def most_informative_features(
        self,
        class_name: str,
        top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Get most discriminative features for a class.

        Args:
            class_name: Target class.
            top_n: Number of features.

        Returns:
            List of (feature, score) tuples.
        """
        if not self._classifier:
            raise RuntimeError("Classifier not trained.")
        return self._classifier.most_informative_features(class_name, top_n)

    def save(self, path: str | Path) -> None:
        """Save the trained model to a JSON file.

        Args:
            path: File path to save to.

        Raises:
            RuntimeError: If the classifier has not been trained.
        """
        if not self._is_trained or not self._vectorizer or not self._classifier:
            raise RuntimeError("Cannot save untrained classifier.")

        model_data = {
            "version": "1.0",
            "vectorizer": self._vectorizer.to_dict(),
            "classifier": self._classifier.to_dict(),
            "vec_kwargs": self._vec_kwargs,
            "cls_kwargs": self._cls_kwargs,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DocumentClassifier":
        """Load a trained model from a JSON file.

        Args:
            path: Path to the saved model file.

        Returns:
            DocumentClassifier instance ready for prediction.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dc = cls(
            vectorizer_kwargs=data.get("vec_kwargs", {}),
            classifier_kwargs=data.get("cls_kwargs", {}),
        )
        dc._vectorizer = TfidfVectorizer.from_dict(data["vectorizer"])
        dc._classifier = NaiveBayesClassifier.from_dict(data["classifier"])
        dc._is_trained = True
        return dc
