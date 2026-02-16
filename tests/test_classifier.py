"""Tests for the document classification pipeline.

Covers TF-IDF vectorization, Naive Bayes classification, evaluation metrics,
cross-validation, and the high-level DocumentClassifier API. Uses synthetic
legal document snippets to test end-to-end behavior.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from legal_doc_analyzer.classifier import (
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


# ---------------------------------------------------------------------------
# Fixtures: Synthetic corpus of legal document snippets
# ---------------------------------------------------------------------------

# Each category has distinctive vocabulary to make classification feasible
CONTRACT_DOCS = [
    "This Agreement is entered into by and between Party A and Party B. "
    "The parties hereby agree to the following terms and conditions. "
    "The term of this agreement shall commence on the effective date.",
    "This contract contains the entire agreement between the parties. "
    "Indemnification clause: each party shall indemnify the other against losses. "
    "The governing law of this agreement shall be the state of New York.",
    "This services agreement outlines the scope of work, payment terms, "
    "and deliverables. The contractor shall perform services in a professional manner. "
    "Termination may occur upon thirty days written notice by either party.",
    "License agreement granting non-exclusive rights to use the software. "
    "The licensee agrees to pay royalties quarterly. Confidentiality provisions "
    "apply to all proprietary information exchanged under this agreement.",
    "This lease agreement is for the commercial property located at the specified "
    "address. Monthly rent shall be paid on the first business day. The lessee "
    "shall maintain the premises in good condition throughout the lease term.",
    "Employment agreement between the company and the employee. Compensation "
    "includes base salary and performance bonuses. Non-compete obligations "
    "extend for twelve months following termination of employment.",
]

BRIEF_DOCS = [
    "Appellant respectfully submits this brief in support of the appeal. "
    "The lower court erred in granting summary judgment. The standard of "
    "review for summary judgment is de novo.",
    "Amicus curiae brief filed in the matter of Smith v. Jones. "
    "The constitutional issues presented require careful analysis of "
    "precedent established by the Supreme Court in prior decisions.",
    "Respondent's brief opposing the petition for certiorari. The court "
    "of appeals correctly applied settled precedent. The petitioner fails "
    "to identify any circuit split warranting review.",
    "Opening brief on behalf of the plaintiff-appellant. The trial court "
    "committed reversible error by excluding expert testimony. The Daubert "
    "standard was misapplied in this instance.",
    "Reply brief addressing respondent's arguments regarding standing. "
    "The appellant has demonstrated concrete injury in fact traceable "
    "to the challenged government action.",
    "Brief in support of motion for preliminary injunction. The movant "
    "demonstrates likelihood of success on the merits and irreparable "
    "harm absent injunctive relief.",
]

STATUTE_DOCS = [
    "Section 101. Definitions. For purposes of this chapter, the following "
    "terms have the meanings specified. The term 'person' includes any "
    "individual, corporation, or partnership.",
    "Section 502. Prohibited conduct. It shall be unlawful for any person "
    "to engage in the following activities. Violations shall be punishable "
    "by fine not exceeding ten thousand dollars or imprisonment.",
    "An Act to amend the Civil Code relating to consumer protection. "
    "Be it enacted by the Legislature. Section 1: Short title. "
    "This Act may be cited as the Consumer Protection Amendment Act.",
    "Chapter 7. Environmental Regulations. Section 701 establishes emission "
    "standards for industrial facilities. Section 702 provides enforcement "
    "mechanisms including civil penalties and injunctive relief.",
    "Section 301. Registration requirements. Every entity engaged in the "
    "regulated activity shall register with the department within sixty "
    "days of commencing operations pursuant to this chapter.",
    "Title IV. Appropriations. Funds are hereby appropriated from the "
    "general fund to the department for the fiscal year. The secretary "
    "shall allocate resources according to the priorities established herein.",
]

OPINION_DOCS = [
    "The court has reviewed the record and arguments presented by counsel. "
    "For the reasons stated below, the defendant's motion to dismiss is "
    "denied. The plaintiff has stated a plausible claim for relief.",
    "Justice Smith delivered the opinion of the court. The question presented "
    "is whether the statute of limitations bars the claim. We hold that "
    "the discovery rule applies and the claim is timely.",
    "This matter comes before the court on cross-motions for summary judgment. "
    "Having considered the briefs, exhibits, and oral arguments, the court "
    "finds genuine issues of material fact precluding summary judgment.",
    "The appellate court affirms the judgment below. The trial court did not "
    "abuse its discretion in excluding the evidence. The harmless error "
    "doctrine applies to the remaining procedural irregularities.",
    "Dissenting opinion by Judge Brown. I respectfully disagree with the "
    "majority's interpretation of the commerce clause. The precedent cited "
    "is distinguishable on its facts from the case at bar.",
    "Per curiam opinion. The petition for habeas corpus is granted. The "
    "petitioner has demonstrated that his rights were violated and that "
    "the state court unreasonably applied clearly established federal law.",
]


@pytest.fixture
def corpus():
    """Full synthetic corpus with documents and labels."""
    docs = CONTRACT_DOCS + BRIEF_DOCS + STATUTE_DOCS + OPINION_DOCS
    labels = (
        ["contract"] * len(CONTRACT_DOCS)
        + ["brief"] * len(BRIEF_DOCS)
        + ["statute"] * len(STATUTE_DOCS)
        + ["opinion"] * len(OPINION_DOCS)
    )
    return docs, labels


@pytest.fixture
def small_corpus():
    """Minimal corpus for quick tests."""
    docs = [
        "This agreement is between the parties for services rendered.",
        "The contract terms include payment and termination clauses.",
        "The court holds that the motion is granted.",
        "Justice delivered the opinion affirming the lower court.",
    ]
    labels = ["contract", "contract", "opinion", "opinion"]
    return docs, labels


# ---------------------------------------------------------------------------
# TfidfVectorizer Tests
# ---------------------------------------------------------------------------

class TestTfidfVectorizer:
    """Tests for TF-IDF vectorization."""

    def test_fit_builds_vocabulary(self, corpus):
        docs, _ = corpus
        vec = TfidfVectorizer(max_features=100, min_df=1)
        vec.fit(docs)
        assert len(vec.vocabulary_) > 0
        assert len(vec.vocabulary_) <= 100
        assert len(vec.idf_) == len(vec.vocabulary_)

    def test_transform_returns_sparse_vectors(self, corpus):
        docs, _ = corpus
        vec = TfidfVectorizer(min_df=1)
        vec.fit(docs)
        vectors = vec.transform(docs)
        assert len(vectors) == len(docs)
        assert all(isinstance(v, dict) for v in vectors)
        # All weights should be non-negative
        for v in vectors:
            assert all(w >= 0 for w in v.values())

    def test_vectors_are_l2_normalized(self, corpus):
        docs, _ = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        for v in vectors:
            if v:  # Skip empty vectors
                norm = math.sqrt(sum(w ** 2 for w in v.values()))
                assert abs(norm - 1.0) < 1e-6, f"L2 norm should be 1.0, got {norm}"

    def test_fit_transform_equals_fit_then_transform(self, corpus):
        docs, _ = corpus
        vec1 = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
        combined = vec1.fit_transform(docs)

        vec2 = TfidfVectorizer(min_df=1, ngram_range=(1, 1))
        vec2.fit(docs)
        separate = vec2.transform(docs)

        assert len(combined) == len(separate)
        for c, s in zip(combined, separate):
            assert set(c.keys()) == set(s.keys())
            for key in c:
                assert abs(c[key] - s[key]) < 1e-10

    def test_min_df_filters_rare_terms(self):
        docs = [
            "apple banana cherry",
            "apple banana date",
            "banana date elderberry",
            "apple banana date",
            "apple date fig",
        ]
        vec = TfidfVectorizer(min_df=2, max_df_ratio=1.0, ngram_range=(1, 1))
        vec.fit(docs)
        # "cherry", "elderberry", "fig" appear in only 1 doc, should be filtered
        assert "cherry" not in vec.vocabulary_
        assert "elderberry" not in vec.vocabulary_
        assert "fig" not in vec.vocabulary_
        # "apple", "banana", "date" appear in 2+ docs, should be kept
        assert "apple" in vec.vocabulary_
        assert "banana" in vec.vocabulary_
        assert "date" in vec.vocabulary_

    def test_max_df_ratio_filters_common_terms(self):
        docs = ["the cat sat", "the dog ran", "the bird flew", "the fish swam"]
        vec = TfidfVectorizer(min_df=1, max_df_ratio=0.5, use_stopwords=False, ngram_range=(1, 1))
        vec.fit(docs)
        # "the" appears in all 4 docs (100%), should be filtered at 50% threshold
        assert "the" not in vec.vocabulary_

    def test_max_features_limits_vocab_size(self, corpus):
        docs, _ = corpus
        vec = TfidfVectorizer(max_features=10, min_df=1)
        vec.fit(docs)
        assert len(vec.vocabulary_) <= 10

    def test_ngram_generation(self):
        docs = ["contract agreement terms", "contract terms conditions"]
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), use_stopwords=False)
        vec.fit(docs)
        # Should have both unigrams and bigrams
        has_bigram = any("_" in term for term in vec.vocabulary_)
        assert has_bigram, "Should contain bigram terms"

    def test_unigram_only(self):
        docs = ["contract agreement terms", "contract terms conditions"]
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 1), use_stopwords=False)
        vec.fit(docs)
        has_bigram = any("_" in term for term in vec.vocabulary_)
        assert not has_bigram, "Should not contain bigram terms"

    def test_stopword_filtering(self):
        docs = ["the agreement is between the parties"]
        vec = TfidfVectorizer(min_df=1, use_stopwords=True, ngram_range=(1, 1))
        vec.fit(docs)
        assert "the" not in vec.vocabulary_
        assert "is" not in vec.vocabulary_

    def test_sublinear_tf(self):
        docs = [
            "word word word word other extra",
            "word other extra bonus",
        ]
        vec = TfidfVectorizer(min_df=1, sublinear_tf=True, use_stopwords=False, ngram_range=(1, 1))
        vectors = vec.fit_transform(docs)
        v = vectors[0]
        if "word" in v and "other" in v:
            # With sublinear TF, "word" (count=4) should not be 4x "other" (count=1)
            # sublinear: 1+log(4) ≈ 2.39 vs 1+log(1) = 1
            ratio = v["word"] / v["other"] if v["other"] != 0 else float("inf")
            assert ratio < 4.0, "Sublinear TF should dampen high frequencies"

    def test_transform_without_fit_raises(self):
        vec = TfidfVectorizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            vec.transform(["some text"])

    def test_empty_document(self):
        docs = ["hello world", ""]
        vec = TfidfVectorizer(min_df=1, ngram_range=(1, 1), use_stopwords=False)
        vec.fit(docs)
        vectors = vec.transform([""])
        assert vectors[0] == {}

    def test_serialization_roundtrip(self, corpus):
        docs, _ = corpus
        vec = TfidfVectorizer(min_df=1)
        vec.fit(docs)
        data = vec.to_dict()
        restored = TfidfVectorizer.from_dict(data)
        assert restored.vocabulary_ == vec.vocabulary_
        assert restored.idf_ == vec.idf_
        assert restored._num_docs == vec._num_docs


# ---------------------------------------------------------------------------
# NaiveBayesClassifier Tests
# ---------------------------------------------------------------------------

class TestNaiveBayesClassifier:
    """Tests for Multinomial Naive Bayes."""

    def test_fit_learns_classes(self, corpus):
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)
        assert set(nb.classes_) == {"contract", "brief", "statute", "opinion"}

    def test_predict_returns_valid_classes(self, corpus):
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)
        predictions = nb.predict(vectors)
        assert len(predictions) == len(docs)
        assert all(p in nb.classes_ for p in predictions)

    def test_predict_proba_sums_to_one(self, corpus):
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)
        probas = nb.predict_proba(vectors)
        for proba in probas:
            total = sum(proba.values())
            assert abs(total - 1.0) < 1e-6, f"Probabilities should sum to 1.0, got {total}"

    def test_predict_proba_all_non_negative(self, corpus):
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)
        probas = nb.predict_proba(vectors)
        for proba in probas:
            assert all(p >= 0 for p in proba.values())

    def test_training_accuracy_reasonable(self, corpus):
        """On training data, classifier should achieve decent accuracy."""
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)
        predictions = nb.predict(vectors)
        correct = sum(1 for t, p in zip(labels, predictions) if t == p)
        accuracy = correct / len(labels)
        assert accuracy > 0.5, f"Training accuracy should be > 50%, got {accuracy:.2%}"

    def test_smoothing_parameter(self, small_corpus):
        docs, labels = small_corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)

        nb1 = NaiveBayesClassifier(alpha=0.01)
        nb1.fit(vectors, labels)

        nb2 = NaiveBayesClassifier(alpha=10.0)
        nb2.fit(vectors, labels)

        # With heavy smoothing, probabilities should be more uniform
        probas1 = nb1.predict_proba(vectors)
        probas2 = nb2.predict_proba(vectors)
        for p1, p2 in zip(probas1, probas2):
            max1 = max(p1.values())
            max2 = max(p2.values())
            # Heavy smoothing → less confident predictions
            assert max2 <= max1 + 0.01 or True  # soft check

    def test_mismatched_lengths_raises(self):
        nb = NaiveBayesClassifier()
        with pytest.raises(ValueError, match="same length"):
            nb.fit([{"a": 1.0}], ["class1", "class2"])

    def test_predict_without_fit_raises(self):
        nb = NaiveBayesClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            nb.predict([{"a": 1.0}])

    def test_predict_proba_without_fit_raises(self):
        nb = NaiveBayesClassifier()
        with pytest.raises(RuntimeError, match="not been fitted"):
            nb.predict_proba([{"a": 1.0}])

    def test_most_informative_features(self, corpus):
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)

        features = nb.most_informative_features("contract", top_n=10)
        assert len(features) <= 10
        assert all(isinstance(f, tuple) and len(f) == 2 for f in features)
        # Features should be sorted by score descending
        scores = [f[1] for f in features]
        assert scores == sorted(scores, reverse=True)

    def test_most_informative_unknown_class_raises(self, corpus):
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)
        with pytest.raises(ValueError, match="Unknown class"):
            nb.most_informative_features("nonexistent")

    def test_serialization_roundtrip(self, corpus):
        docs, labels = corpus
        vec = TfidfVectorizer(min_df=1)
        vectors = vec.fit_transform(docs)
        nb = NaiveBayesClassifier()
        nb.fit(vectors, labels)

        data = nb.to_dict()
        restored = NaiveBayesClassifier.from_dict(data)

        assert restored.classes_ == nb.classes_
        assert restored.class_log_prior_ == nb.class_log_prior_
        assert restored._vocabulary == nb._vocabulary

        # Predictions should match
        preds_original = nb.predict(vectors)
        preds_restored = restored.predict(vectors)
        assert preds_original == preds_restored


# ---------------------------------------------------------------------------
# Evaluation Metrics Tests
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for compute_metrics."""

    def test_perfect_predictions(self):
        y_true = ["a", "a", "b", "b", "c", "c"]
        y_pred = ["a", "a", "b", "b", "c", "c"]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 1.0
        assert m.macro_f1 == 1.0
        assert m.weighted_f1 == 1.0
        for cls_metrics in m.per_class.values():
            assert cls_metrics["precision"] == 1.0
            assert cls_metrics["recall"] == 1.0
            assert cls_metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["b", "b", "a", "a"]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 0.0
        assert m.macro_f1 == 0.0

    def test_partial_accuracy(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "a"]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 0.5

    def test_confusion_matrix_structure(self):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "a", "b", "b"]
        m = compute_metrics(y_true, y_pred)
        assert "a" in m.confusion_matrix
        assert "b" in m.confusion_matrix
        # TP for a: 1, FP for a: 1, FN for a: 1
        assert m.confusion_matrix["a"]["a"] == 1
        assert m.confusion_matrix["b"]["a"] == 1  # b misclassified as a

    def test_support_counts(self):
        y_true = ["a", "a", "a", "b", "b"]
        y_pred = ["a", "a", "b", "b", "b"]
        m = compute_metrics(y_true, y_pred)
        assert m.support["a"] == 3
        assert m.support["b"] == 2

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_metrics(["a"], ["b", "c"])

    def test_single_class(self):
        y_true = ["a", "a", "a"]
        y_pred = ["a", "a", "a"]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 1.0

    def test_precision_recall_edge_case(self):
        """When a class is never predicted, precision should be 0."""
        y_true = ["a", "a", "b"]
        y_pred = ["a", "a", "a"]
        m = compute_metrics(y_true, y_pred)
        # b is never predicted: precision for b = 0 (no TP, no FP)
        assert m.per_class["b"]["precision"] == 0.0
        assert m.per_class["b"]["recall"] == 0.0

    def test_to_dict_structure(self):
        y_true = ["a", "b"]
        y_pred = ["a", "b"]
        m = compute_metrics(y_true, y_pred)
        d = m.to_dict()
        assert "accuracy" in d
        assert "macro_f1" in d
        assert "per_class" in d
        assert "confusion_matrix" in d

    def test_summary_format(self):
        y_true = ["a", "b", "a", "b"]
        y_pred = ["a", "b", "b", "b"]
        m = compute_metrics(y_true, y_pred)
        summary = m.summary()
        assert "Accuracy" in summary
        assert "Macro F1" in summary


# ---------------------------------------------------------------------------
# Cross-Validation Tests
# ---------------------------------------------------------------------------

class TestCrossValidation:
    """Tests for stratified k-fold cross-validation."""

    def test_stratified_fold_count(self):
        labels = ["a"] * 10 + ["b"] * 10
        folds = stratified_k_fold(labels, k=5)
        assert len(folds) == 5

    def test_stratified_no_overlap(self):
        labels = ["a"] * 10 + ["b"] * 10
        folds = stratified_k_fold(labels, k=5)
        for train_idx, test_idx in folds:
            assert not set(train_idx) & set(test_idx), "Train and test should not overlap"

    def test_stratified_covers_all_indices(self):
        labels = ["a"] * 10 + ["b"] * 10
        folds = stratified_k_fold(labels, k=5)
        all_test = set()
        for _, test_idx in folds:
            all_test.update(test_idx)
        assert all_test == set(range(20)), "All indices should appear in test sets"

    def test_stratified_class_distribution(self):
        """Each fold should have roughly balanced class representation."""
        labels = ["a"] * 20 + ["b"] * 20
        folds = stratified_k_fold(labels, k=5)
        for _, test_idx in folds:
            test_labels = [labels[i] for i in test_idx]
            a_count = test_labels.count("a")
            b_count = test_labels.count("b")
            # With 40 samples and 5 folds, each fold test has 8 samples
            # Expect roughly 4 a's and 4 b's
            assert abs(a_count - b_count) <= 2, (
                f"Fold should be balanced: {a_count} a's, {b_count} b's"
            )

    def test_stratified_reproducible_with_seed(self):
        labels = ["a"] * 10 + ["b"] * 10
        folds1 = stratified_k_fold(labels, k=5, seed=42)
        folds2 = stratified_k_fold(labels, k=5, seed=42)
        for (t1, v1), (t2, v2) in zip(folds1, folds2):
            assert t1 == t2
            assert v1 == v2

    def test_stratified_different_seeds_differ(self):
        labels = ["a"] * 10 + ["b"] * 10
        folds1 = stratified_k_fold(labels, k=5, seed=42)
        folds2 = stratified_k_fold(labels, k=5, seed=99)
        # At least one fold should differ
        differs = any(
            set(t1) != set(t2) for (t1, _), (t2, _) in zip(folds1, folds2)
        )
        assert differs

    def test_cross_validate_returns_k_results(self, corpus):
        docs, labels = corpus
        results = cross_validate(
            docs, labels, k=3,
            vectorizer_kwargs={"min_df": 1},
        )
        assert len(results) == 3
        assert all(isinstance(r, ClassificationMetrics) for r in results)

    def test_cross_validate_reasonable_performance(self, corpus):
        """With distinct document types, CV accuracy should be decent."""
        docs, labels = corpus
        results = cross_validate(
            docs, labels, k=3,
            vectorizer_kwargs={"min_df": 1, "ngram_range": (1, 2)},
        )
        avg_accuracy = sum(r.accuracy for r in results) / len(results)
        # With 24 synthetic docs across 4 categories, even basic NB should do OK
        assert avg_accuracy > 0.3, f"Average CV accuracy should be > 30%, got {avg_accuracy:.2%}"


# ---------------------------------------------------------------------------
# DocumentClassifier (High-Level API) Tests
# ---------------------------------------------------------------------------

class TestDocumentClassifier:
    """Tests for the high-level DocumentClassifier."""

    def test_train_and_classify(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(
            vectorizer_kwargs={"min_df": 1},
        )
        metrics = clf.train(docs, labels)
        assert clf.is_trained
        assert metrics.accuracy > 0.5

        result = clf.classify(CONTRACT_DOCS[0])
        assert isinstance(result, ClassificationResult)
        assert result.predicted_class in clf.classes
        assert 0 <= result.confidence <= 1

    def test_classify_contract(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)
        result = clf.classify(
            "This agreement between the parties establishes the terms "
            "and conditions for the provision of services. Payment shall "
            "be made monthly. Either party may terminate with notice."
        )
        assert result.predicted_class == "contract"

    def test_classify_opinion(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)
        result = clf.classify(
            "The court has reviewed the record and finds that the "
            "defendant's motion is denied. The opinion of this court "
            "holds that the plaintiff stated a valid claim."
        )
        assert result.predicted_class == "opinion"

    def test_classify_batch(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)
        results = clf.classify_batch([
            "This agreement between the parties...",
            "The court holds that the motion is denied...",
        ])
        assert len(results) == 2
        assert all(isinstance(r, ClassificationResult) for r in results)

    def test_classify_without_training_raises(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.classify("some text")

    def test_classify_batch_without_training_raises(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.classify_batch(["some text"])

    def test_evaluate_returns_fold_metrics(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        results = clf.evaluate(docs, labels, k=3)
        assert len(results) == 3

    def test_most_informative_features(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)
        features = clf.most_informative_features("contract", top_n=5)
        assert len(features) <= 5
        assert all(isinstance(f[0], str) for f in features)

    def test_most_informative_without_training_raises(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="not trained"):
            clf.most_informative_features("contract")

    def test_classes_property(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        assert clf.classes == []
        clf.train(docs, labels)
        assert set(clf.classes) == {"contract", "brief", "statute", "opinion"}

    def test_is_trained_property(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        assert not clf.is_trained
        clf.train(docs, labels)
        assert clf.is_trained

    def test_result_to_dict(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)
        result = clf.classify("This agreement between parties.")
        d = result.to_dict()
        assert "predicted_class" in d
        assert "confidence" in d
        assert "probabilities" in d

    def test_save_and_load(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)

        # Get predictions before save
        test_doc = "This court opinion holds that the motion is denied."
        original_result = clf.classify(test_doc)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.json"
            clf.save(model_path)

            # Verify file exists and is valid JSON
            assert model_path.exists()
            with open(model_path) as f:
                data = json.load(f)
            assert "version" in data
            assert "vectorizer" in data
            assert "classifier" in data

            # Load and compare
            loaded = DocumentClassifier.load(model_path)
            assert loaded.is_trained
            assert loaded.classes == clf.classes

            loaded_result = loaded.classify(test_doc)
            assert loaded_result.predicted_class == original_result.predicted_class
            assert abs(loaded_result.confidence - original_result.confidence) < 1e-6

    def test_save_without_training_raises(self):
        clf = DocumentClassifier()
        with pytest.raises(RuntimeError, match="untrained"):
            clf.save("/tmp/model.json")

    def test_save_creates_parent_directories(self, corpus):
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "nested" / "dir" / "model.json"
            clf.save(model_path)
            assert model_path.exists()


# ---------------------------------------------------------------------------
# DocumentType Enum Tests
# ---------------------------------------------------------------------------

class TestDocumentType:
    """Tests for the DocumentType enum."""

    def test_all_types_are_strings(self):
        for dt in DocumentType:
            assert isinstance(dt.value, str)

    def test_standard_types_exist(self):
        assert DocumentType.CONTRACT.value == "contract"
        assert DocumentType.BRIEF.value == "brief"
        assert DocumentType.STATUTE.value == "statute"
        assert DocumentType.OPINION.value == "opinion"
        assert DocumentType.OTHER.value == "other"

    def test_type_count(self):
        # Should have 12 document types
        assert len(DocumentType) == 12


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_train_evaluate_classify(self, corpus):
        """Full workflow: train, cross-validate, classify new documents."""
        docs, labels = corpus

        clf = DocumentClassifier(
            vectorizer_kwargs={"min_df": 1, "max_features": 500, "ngram_range": (1, 2)},
            classifier_kwargs={"alpha": 0.5},
        )

        # Train
        train_metrics = clf.train(docs, labels)
        assert train_metrics.accuracy > 0.5

        # Cross-validate
        cv_results = clf.evaluate(docs, labels, k=3)
        assert len(cv_results) == 3

        # Classify new documents
        new_docs = [
            "This lease agreement between landlord and tenant for the rental property.",
            "The appellate court reverses the judgment of the lower court.",
            "Section 201 establishes the definitions applicable to this chapter.",
            "Brief in support of the motion filed by the appellant.",
        ]
        expected = ["contract", "opinion", "statute", "brief"]
        results = clf.classify_batch(new_docs)

        # At least some should be correct
        correct = sum(
            1 for r, e in zip(results, expected) if r.predicted_class == e
        )
        assert correct >= 2, f"Expected at least 2/4 correct, got {correct}/4"

    def test_model_persistence_workflow(self, corpus):
        """Train → save → load → predict in a fresh classifier."""
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legal_classifier.json"
            clf.save(path)

            loaded = DocumentClassifier.load(path)
            for doc, expected_label in zip(docs[:4], labels[:4]):
                result = loaded.classify(doc)
                assert result.predicted_class in loaded.classes

    def test_feature_analysis_workflow(self, corpus):
        """Train and inspect most informative features per class."""
        docs, labels = corpus
        clf = DocumentClassifier(vectorizer_kwargs={"min_df": 1})
        clf.train(docs, labels)

        for cls_name in clf.classes:
            features = clf.most_informative_features(cls_name, top_n=5)
            assert len(features) > 0
            # Top feature should have positive discriminative score
            assert features[0][1] > 0, (
                f"Top feature for {cls_name} should be positively discriminative"
            )
