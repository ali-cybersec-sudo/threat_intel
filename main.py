"""main.py
=========

Entry point for the CTI Multi-Agent System.

Usage::

    # Launch the Streamlit UI (default)
    python main.py --ui

    # Run a single query from the command line
    python main.py --query "Analyze IP 8.8.8.8"

    # Ingest reports into the RAG vector store
    python main.py --ingest ./data/raw_reports

    # Show help
    python main.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Logging configuration ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cti_system")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def run_ui() -> None:
    """Launch the Streamlit UI."""
    import subprocess
    ui_path = str(Path(__file__).resolve().parent / "ui" / "app.py")
    url = "http://localhost:8501"
    print(f"\nStarting Streamlit UI. If the browser does not open, visit: {url}\n")
    logger.info("Launching Streamlit UI: %s", ui_path)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", ui_path,
         "--server.address", "localhost",
         "--server.port", "8501",
         "--server.headless", "false",
         "--browser.gatherUsageStats", "false"],
        check=True,
    )


def run_query(query: str) -> None:
    """Execute a single query via the Orchestrator and print results."""
    from core.orchestrator import Orchestrator

    logger.info("Processing query: %s", query)
    orch = Orchestrator()
    result = orch.handle_query(query)

    # Print the markdown report if available, otherwise the summary
    if result.get("markdown"):
        print("\n" + result["markdown"])
    else:
        print(f"\n[{result.get('status', 'unknown')}] {result.get('summary', '')}")

    # Print metadata
    print(f"\n--- Metadata ---")
    print(f"Intent:     {result.get('intent')}")
    print(f"Severity:   {result.get('severity')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Agents:     {result.get('agents_used')}")
    print(f"Time:       {result.get('elapsed_seconds')}s")


def run_ingest(directory: str) -> None:
    """Ingest documents from a directory into the RAG vector store."""
    from config.config_loader import ConfigLoader
    from tools.rag_engine import RAGEngine

    loader = ConfigLoader.instance()
    rag_cfg = {
        "collection_name": loader.settings.get("rag", {}).get("collection_name", "cti_reports"),
        "persist_dir": loader.settings.get("rag", {}).get("persist_dir", "./data/chroma_db"),
    }
    engine = RAGEngine(rag_cfg)
    count = engine.ingest_directory(directory)
    print(f"Ingested {count} chunks from {directory}")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate handler."""
    parser = argparse.ArgumentParser(
        description="CTI Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py --query "Analyze IP 1.2.3.4"\n'
            "  python main.py --ui\n"
            "  python main.py --ingest ./data/raw_reports\n"
        ),
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Streamlit web interface.",
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Run a single CTI query from the command line.",
    )
    parser.add_argument(
        "--ingest", "-i",
        type=str,
        default=None,
        help="Ingest all .txt/.pdf files from the given directory into the RAG store.",
    )

    args = parser.parse_args()

    if args.ingest:
        run_ingest(args.ingest)
    elif args.query:
        run_query(args.query)
    elif args.ui:
        run_ui()
    else:
        # Default: launch UI
        run_ui()


if __name__ == "__main__":
    main()
