"""app.py
=========

Streamlit-based UI for the CTI Multi-Agent System.

Run with::

    streamlit run ui/app.py

Or via ``main.py``::

    python main.py --ui
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import streamlit as st

# Ensure the project root is on sys.path so imports work when running
# ``streamlit run ui/app.py`` from any working directory.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.orchestrator import Orchestrator

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = Orchestrator()
logger = logging.getLogger(__name__)

# ── Streamlit page configuration ────────────────────────────────────────

st.set_page_config(
    page_title="CTI Multi-Agent System",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #333;
    }
    .severity-critical { color: #ff4444; font-weight: 700; }
    .severity-high     { color: #ff8800; font-weight: 700; }
    .severity-medium   { color: #ffcc00; font-weight: 600; }
    .severity-low      { color: #44bb44; font-weight: 600; }
    .severity-info     { color: #4488ff; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ─────────────────────────────────────────

def _init_session() -> None:
    """Initialise Streamlit session-state keys on first run."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = Orchestrator()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


_init_session()


# ── Sidebar ──────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render the sidebar with system info and controls."""
    with st.sidebar:
        st.markdown("### \U0001f6e1\ufe0f CTI System")
        st.caption("Multi-Agent Cyber Threat Intelligence")
        st.divider()

        st.markdown("**Quick Queries**")
        examples = [
            "Analyze IP 8.8.8.8",
            "Check hash 44d88612fea8a8f36de82e1278abb02f",
            "Lookup CVE-2024-3400",
            "Research APT29 campaign",
            "MITRE technique T1059",
            "Analyze domain evil-corp.example.net",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}", use_container_width=True):
                st.session_state.sidebar_query = ex

        st.divider()

        if st.button("\U0001f5d1\ufe0f Clear Session", use_container_width=True):
            st.session_state.orchestrator.clear_session()
            st.session_state.messages.clear()
            st.session_state.last_result = None
            st.rerun()

        st.divider()
        st.markdown("**Session History**")
        history = st.session_state.orchestrator.get_history()
        if history:
            for i, entry in enumerate(reversed(history[-5:])):
                q = entry.get("query", "")[:50]
                st.caption(f"{i + 1}. {q}...")
        else:
            st.caption("No queries yet.")


_render_sidebar()


# ── Main area ────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">\U0001f6e1\ufe0f Cyber Threat Intelligence Analyst</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    "Multi-agent system for IOC analysis, threat research and CTI reporting"
    "</div>",
    unsafe_allow_html=True,
)


# ── Chat history ─────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ───────────────────────────────────────────────────────────

# Accept input from sidebar quick-query buttons or the chat box
sidebar_query = st.session_state.pop("sidebar_query", None)
user_input = sidebar_query or st.chat_input("Enter a CTI query (IP, hash, domain, CVE, MITRE ID, or research topic)...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Agents are working..."):
            try:
                result = st.session_state.orchestrator.route_query(user_input)
                st.session_state.last_result = result
            except Exception as exc:
                logger.exception("Query processing failed.")
                result = {
                    "status": "error",
                    "answer": f"An unexpected error occurred: {exc}",
                }

        # ── Result display ───────────────────────────────────────────
        if result.get("status") == "error":
            st.error(result.get("answer", "Unknown error."))
        else:
            answer = result.get("answer", "")
            report = result.get("report", "")

            if report:
                if answer:
                    st.info(answer)
                
                # Metrics
                col1, col2 = st.columns(2)
                severity = result.get("severity")
                if severity:
                    sev_class = f"severity-{severity.lower()}"
                    col1.markdown(f'Severity: <span class="{sev_class}">{severity}</span>', unsafe_allow_html=True)
                confidence = result.get("confidence")
                if confidence is not None:
                    col2.metric("Confidence", f"{confidence:.0%}")

                # Agents used
                agents = result.get("agents_used", [])
                if agents:
                    st.caption(f"Agents: {' \u2192 '.join(agents)}")
                if result.get("memory_used"):
                    memory_record = result.get("memory_record") or {}
                    st.caption(
                        "Memory used"
                        + (f": {memory_record.get('indicator')} ({memory_record.get('timestamp')})" if memory_record else "")
                    )
                elif result.get("memory_saved"):
                    memory_record = result.get("memory_record") or {}
                    st.caption(
                        "Saved to CTI memory"
                        + (f": {memory_record.get('indicator')}" if memory_record else "")
                    )

                with st.expander("\U0001f4cb Detailed Report", expanded=True):
                    st.markdown(report)
            else:
                if answer:
                    st.markdown(answer)
                    if result.get("memory_used"):
                        memory_record = result.get("memory_record") or {}
                        st.caption(
                            "Memory used"
                            + (f": {memory_record.get('indicator')} ({memory_record.get('timestamp')})" if memory_record else "")
                        )
                else:
                    st.info("Analysis complete.")

        # Store assistant message
        display_text = answer if answer else "Analysis complete."
        st.session_state.messages.append({"role": "assistant", "content": display_text})

if __name__ == "__main__":
    # No direct execution path needed for Streamlit UI.
    # This block is retained for compatibility but does nothing.
    pass
