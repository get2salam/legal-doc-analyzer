"""Shared test fixtures for legal-doc-analyzer tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_contract_path() -> Path:
    """Path to the sample contract text file."""
    return Path(__file__).parent.parent / "examples" / "sample_contract.txt"


@pytest.fixture
def sample_contract_text(sample_contract_path: Path) -> str:
    """Full text of the sample contract."""
    return sample_contract_path.read_text(encoding="utf-8")


@pytest.fixture
def short_legal_text() -> str:
    """A short legal text snippet for targeted tests."""
    return (
        "This Agreement is entered into as of March 10, 2024, "
        'by and between Alpha Corp. ("Client") and Beta Services LLC ("Provider").\n\n'
        "1. CONFIDENTIALITY\n"
        "The Provider shall maintain the confidentiality of all proprietary information "
        "and trade secrets disclosed by the Client. The receiving party shall not disclose "
        "such confidential information to any third party.\n\n"
        "2. PAYMENT\n"
        "The Client shall pay the Provider $50,000 upon execution of this Agreement "
        "and $25,000 upon completion. Late payments shall accrue interest at 1% per month. "
        "Invoice is due within net 30 days.\n\n"
        "3. GOVERNING LAW\n"
        "This Agreement shall be governed by the laws of the State of New York.\n\n"
        "4. TERMINATION\n"
        "Either party may terminate this Agreement upon thirty (30) days written notice "
        "if the other party commits a material breach.\n\n"
        "5. LIMITATION OF LIABILITY\n"
        "In no event shall either party be liable for any indirect, incidental, "
        "or consequential damages. The aggregate liability shall not exceed "
        "the total fees paid under this Agreement.\n"
    )


@pytest.fixture
def minimal_text() -> str:
    """Minimal text with almost no legal content (for edge-case testing)."""
    return "Hello world. This is a simple document with no legal clauses."


@pytest.fixture
def tmp_text_file(tmp_path: Path, short_legal_text: str) -> Path:
    """Create a temporary text file with legal content."""
    file = tmp_path / "test_contract.txt"
    file.write_text(short_legal_text, encoding="utf-8")
    return file
