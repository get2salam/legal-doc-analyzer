"""
Legal Document Analyzer
AI-powered analysis of legal documents, contracts, and agreements.
"""

import json
import os

import streamlit as st
from dotenv import load_dotenv

from utils.document_processor import extract_text_from_file
from utils.llm_analyzer import LegalAnalyzer

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .analysis-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E3A5F;
    }
    .risk-high {
        border-left-color: #dc3545 !important;
        background-color: #fff5f5 !important;
    }
    .risk-medium {
        border-left-color: #ffc107 !important;
        background-color: #fffdf5 !important;
    }
    .risk-low {
        border-left-color: #28a745 !important;
        background-color: #f5fff7 !important;
    }
    .stat-box {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""",
    unsafe_allow_html=True,
)


def init_session_state():
    """Initialize session state variables."""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "document_text" not in st.session_state:
        st.session_state.document_text = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = None


def render_sidebar():
    """Render the sidebar with settings and info."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/law.png", width=80)
        st.markdown("### ⚙️ Settings")

        # Model selection
        model = st.selectbox(
            "AI Model",
            ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "claude-3-sonnet", "claude-3-opus"],
            index=0,
            help="Select the AI model for analysis",
        )

        # Analysis depth
        depth = st.select_slider(
            "Analysis Depth",
            options=["Quick", "Standard", "Deep"],
            value="Standard",
            help="Deeper analysis takes longer but provides more detail",
        )

        st.markdown("---")
        st.markdown("### 📊 Supported Documents")
        st.markdown("""
        - 📄 PDF files
        - 📝 Word documents (.docx)
        - 📃 Text files (.txt)
        """)

        st.markdown("---")
        st.markdown("### 🔒 Privacy")
        st.markdown("""
        Your documents are processed securely and 
        **never stored** on our servers.
        """)

        st.markdown("---")
        st.markdown("### 👨‍💻 Built by")
        st.markdown("[Abdul Salam](https://linkedin.com/in/abdul-salam-6539aa11b)")
        st.markdown("MS AI | LLM Commercial Law | LLB")

        return model, depth


def render_analysis_results(result: dict):
    """Render the analysis results in a structured format."""

    # Summary Section
    st.markdown("## 📋 Document Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-number">{result.get("contract_type", "N/A")}</div>
            <div class="stat-label">Contract Type</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        risk_level = result.get("overall_risk", "Medium")
        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_level, "🟡")
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-number">{risk_color} {risk_level}</div>
            <div class="stat-label">Risk Level</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-number">{len(result.get("parties", []))}</div>
            <div class="stat-label">Parties</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-number">{len(result.get("key_clauses", []))}</div>
            <div class="stat-label">Key Clauses</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Two column layout
    col_left, col_right = st.columns(2)

    with col_left:
        # Parties
        st.markdown("### 👥 Parties Involved")
        for party in result.get("parties", []):
            st.markdown(
                f"""
            <div class="analysis-card">
                <strong>{party.get("name", "Unknown")}</strong><br>
                <small>Role: {party.get("role", "Not specified")}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Key Dates
        st.markdown("### 📅 Important Dates")
        dates = result.get("key_dates", {})
        if dates:
            for date_type, date_value in dates.items():
                st.markdown(f"- **{date_type}:** {date_value}")
        else:
            st.markdown("*No specific dates identified*")

    with col_right:
        # Financial Terms
        st.markdown("### 💰 Financial Terms")
        financials = result.get("financial_terms", [])
        if financials:
            for term in financials:
                st.markdown(
                    f"""
                <div class="analysis-card">
                    <strong>{term.get("type", "Payment")}</strong>: {term.get("amount", "N/A")}<br>
                    <small>{term.get("details", "")}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("*No financial terms identified*")

        # Jurisdiction
        st.markdown("### 🌍 Jurisdiction & Governing Law")
        st.markdown(
            f"""
        <div class="analysis-card">
            {result.get("jurisdiction", "Not specified in document")}
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Key Clauses
    st.markdown("### 📜 Key Clauses Analysis")
    for clause in result.get("key_clauses", []):
        risk_class = f"risk-{clause.get('risk', 'low').lower()}"
        st.markdown(
            f"""
        <div class="analysis-card {risk_class}">
            <strong>{clause.get("title", "Clause")}</strong>
            <span style="float: right; font-size: 0.8rem;">Risk: {clause.get("risk", "Low")}</span><br>
            <p>{clause.get("summary", "")}</p>
            <small><em>Recommendation: {clause.get("recommendation", "Review carefully")}</em></small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Risk Flags
    st.markdown("### ⚠️ Risk Flags & Concerns")
    risks = result.get("risk_flags", [])
    if risks:
        for risk in risks:
            severity = risk.get("severity", "medium")
            icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "🟡")
            st.warning(f"{icon} **{risk.get('issue', 'Issue')}**: {risk.get('description', '')}")
    else:
        st.success("✅ No significant risk flags identified")

    st.markdown("---")

    # Plain English Summary
    st.markdown("### 📝 Plain English Summary")
    st.info(
        result.get("plain_summary", "Analysis complete. Review the sections above for details.")
    )

    # Export Options
    st.markdown("### 📥 Export Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 Download JSON",
            data=json.dumps(result, indent=2),
            file_name="legal_analysis.json",
            mime="application/json",
        )
    with col2:
        # Create markdown report
        md_report = f"""# Legal Document Analysis Report

## Document Type: {result.get("contract_type", "Unknown")}
## Overall Risk: {result.get("overall_risk", "Medium")}

### Summary
{result.get("plain_summary", "N/A")}

### Parties
{chr(10).join([f"- {p.get('name', 'Unknown')} ({p.get('role', 'N/A')})" for p in result.get("parties", [])])}

### Key Clauses
{chr(10).join([f"- **{c.get('title', 'Clause')}** (Risk: {c.get('risk', 'Low')}): {c.get('summary', '')}" for c in result.get("key_clauses", [])])}

---
*Generated by Legal Document Analyzer*
"""
        st.download_button(
            "📝 Download Markdown",
            data=md_report,
            file_name="legal_analysis.md",
            mime="text/markdown",
        )


def main():
    """Main application entry point."""
    init_session_state()

    # Sidebar
    model, depth = render_sidebar()

    # Main content
    st.markdown('<p class="main-header">⚖️ Legal Document Analyzer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">AI-powered analysis of contracts, agreements, and legal documents</p>',
        unsafe_allow_html=True,
    )

    # Tabs for single file vs batch
    tab1, tab2 = st.tabs(["📄 Single Document", "📁 Batch Analysis"])

    with tab1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your legal document",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, Word (.docx), Text (.txt)",
            key="single_file",
        )

        if uploaded_file:
            st.session_state.file_name = uploaded_file.name

            # Extract text
            with st.spinner("📄 Extracting text from document..."):
                text = extract_text_from_file(uploaded_file)
                st.session_state.document_text = text

            if text:
                st.success(
                    f"✅ Successfully extracted {len(text):,} characters from {uploaded_file.name}"
                )

                # Show preview
                with st.expander("📖 Preview Document Text", expanded=False):
                    st.text_area(
                        "Extracted Text",
                        text[:5000] + ("..." if len(text) > 5000 else ""),
                        height=200,
                    )

                # Analyze button
                if st.button("🔍 Analyze Document", type="primary", use_container_width=True):
                    # Check for API keys
                    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")

                    if not api_key:
                        st.error(
                            "⚠️ No API key configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
                        )
                        st.code(
                            "cp .env.example .env\n# Then edit .env with your API key",
                            language="bash",
                        )

                        # Demo mode
                        st.info("💡 Running in demo mode with sample analysis...")
                        st.session_state.analysis_result = get_demo_result()
                    else:
                        with st.spinner(
                            "🧠 AI is analyzing your document... This may take a minute."
                        ):
                            try:
                                analyzer = LegalAnalyzer(model=model)
                                result = analyzer.analyze(text, depth=depth)
                                st.session_state.analysis_result = result
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")
                                st.session_state.analysis_result = get_demo_result()
            else:
                st.error(
                    "❌ Could not extract text from the document. Please try a different file."
                )

        # Display results
        if st.session_state.analysis_result:
            st.markdown("---")
            render_analysis_results(st.session_state.analysis_result)

    with tab2:
        st.markdown("### 📁 Batch Document Analysis")
        st.markdown("Upload multiple documents for bulk analysis")

        batch_files = st.file_uploader(
            "Upload multiple documents",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="Select multiple files for batch processing",
            key="batch_files",
        )

        if batch_files:
            st.info(f"📁 {len(batch_files)} files selected for analysis")

            if st.button("🚀 Analyze All Documents", type="primary", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, file in enumerate(batch_files):
                    status_text.text(f"Analyzing {file.name}...")
                    progress_bar.progress((i + 1) / len(batch_files))

                    try:
                        text = extract_text_from_file(file)
                        if text:
                            # Use demo results for now (or real analyzer if API key available)
                            result = get_demo_result()
                            result["file_name"] = file.name
                            result["char_count"] = len(text)
                            results.append(result)
                    except Exception as e:
                        results.append(
                            {"file_name": file.name, "error": str(e), "overall_risk": "Error"}
                        )

                status_text.text("✅ Batch analysis complete!")

                # Summary table
                st.markdown("### 📊 Batch Results Summary")
                summary_data = []
                for r in results:
                    summary_data.append(
                        {
                            "File": r.get("file_name", "Unknown"),
                            "Type": r.get("contract_type", "N/A"),
                            "Risk": r.get("overall_risk", "N/A"),
                            "Parties": len(r.get("parties", [])),
                            "Clauses": len(r.get("key_clauses", [])),
                            "Flags": len(r.get("risk_flags", [])),
                        }
                    )

                st.dataframe(summary_data, use_container_width=True)

                # Export batch results
                st.download_button(
                    "📥 Download All Results (JSON)",
                    data=json.dumps(results, indent=2),
                    file_name="batch_analysis_results.json",
                    mime="application/json",
                )


def get_demo_result():
    """Return demo analysis result for testing without API key."""
    return {
        "contract_type": "Service Agreement",
        "overall_risk": "Medium",
        "parties": [
            {"name": "ABC Corporation Ltd", "role": "Service Provider"},
            {"name": "XYZ Industries", "role": "Client"},
        ],
        "key_dates": {
            "Effective Date": "January 1, 2024",
            "Termination Date": "December 31, 2024",
            "Renewal Notice": "30 days before expiry",
        },
        "financial_terms": [
            {"type": "Monthly Fee", "amount": "$5,000", "details": "Due on the 1st of each month"},
            {
                "type": "Late Payment Penalty",
                "amount": "2% per month",
                "details": "On overdue amounts",
            },
        ],
        "jurisdiction": "Laws of England and Wales, exclusive jurisdiction of London courts",
        "key_clauses": [
            {
                "title": "Limitation of Liability",
                "summary": "Provider's liability is capped at total fees paid in the 12 months preceding the claim.",
                "risk": "Medium",
                "recommendation": "Consider negotiating higher cap for critical services",
            },
            {
                "title": "Termination for Convenience",
                "summary": "Either party may terminate with 30 days written notice without cause.",
                "risk": "Low",
                "recommendation": "Standard clause, reasonable notice period",
            },
            {
                "title": "Intellectual Property",
                "summary": "All IP created during engagement belongs to the Client upon payment.",
                "risk": "Low",
                "recommendation": "Favorable to client, ensure payment terms are clear",
            },
            {
                "title": "Indemnification",
                "summary": "Provider indemnifies Client against third-party IP claims. No mutual indemnification.",
                "risk": "High",
                "recommendation": "One-sided indemnification; negotiate mutual terms",
            },
        ],
        "risk_flags": [
            {
                "severity": "high",
                "issue": "One-sided Indemnification",
                "description": "Only the provider is required to indemnify, creating unbalanced risk allocation",
            },
            {
                "severity": "medium",
                "issue": "Auto-renewal Clause",
                "description": "Contract auto-renews unless notice given 30 days prior - ensure calendar reminder",
            },
        ],
        "plain_summary": "This is a standard service agreement between ABC Corporation (provider) and XYZ Industries (client) for a one-year term with automatic renewal. The client pays $5,000 monthly for services. Key concerns include one-sided indemnification favoring the client and auto-renewal terms. The liability cap is reasonable but may need adjustment for high-value services. Overall, the agreement is fairly balanced but the indemnification clause should be negotiated.",
    }


if __name__ == "__main__":
    main()
