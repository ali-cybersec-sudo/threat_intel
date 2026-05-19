# Cyber Threat Intelligence Analyst

Multi-agent Streamlit system for IOC analysis, threat research, persistent memory, and CTI reporting.

## What It Does

- Investigates IPs, domains, hashes, CVEs, MITRE ATT&CK techniques, and threat research topics.
- Routes each query through specialized agents instead of one monolithic prompt.
- Stores structured CTI memory under `data/memory/` so follow-up questions can refer to the previous IOC.
- Persists lightweight agent memory under `data/vector_db/` and repeated OSINT answers under `data/cache/`.
- Uses live tools where configured: web search, AbuseIPDB, VirusTotal, NVD, DNS/WHOIS, RAG, and CAG cache.
- Produces markdown CTI reports with IOC evidence, severity, ATT&CK mapping, limitations, and recommendations.
- Supports English and Arabic CTI follow-ups such as `is this IP malicious?` and `هل الاي بي ده ضار؟`.

## Course Requirement Mapping

- **Multi-Agent System:** `OSINTAgent`, `AnalystAgent`, and `ReporterAgent` are coordinated by `Orchestrator` and planned by `Router`.
- **Advanced Memory:** `CTIMemory` persists structured investigation records and resolves follow-up questions from previous indicators.
- **Tool Integration:** Web search, RAG retrieval, CAG caching, enrichment APIs, and LLM generation are integrated through `tools/`.
- **User Interface:** Streamlit chat interface in `ui/app.py`.
- **Bonus:** Arabic/English follow-up support plus input/output guardrails.

## Run

```bash
python main.py --ui
```

or one CLI query:

```bash
python main.py --query "search ip 185.164.81.156"
```

## Demo Hardening Check

Run the offline checklist before the live presentation:

```bash
python scripts/demo_hardening_check.py
```

The script verifies routing, durable CTI memory, Arabic follow-ups, prompt-injection blocking, output redaction, persistent CAG cache, persistent agent memory, and safe ATT&CK report formatting without calling live APIs.

## Configuration

Copy `.env.example` to `.env` and add the API keys you want to use. The app can still run in degraded mode when some providers are unavailable, but reports clearly show limitations instead of inventing evidence.
