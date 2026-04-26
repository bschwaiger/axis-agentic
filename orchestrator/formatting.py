"""Shared formatting helpers for demo-friendly terminal output.

Provider-agnostic. The engine's provider_name is passed in for any
"called by X" / "X says" rendering, so the same formatter works for
Anthropic, OpenAI, Nemotron, Ollama, etc.
"""
from __future__ import annotations


# ------------------------------------------------------------------
# Tool call summaries (one-line, human-readable)
# ------------------------------------------------------------------

def format_tool_call(fn_name: str, fn_args: dict) -> str:
    """Format a tool call as a concise one-liner."""
    key_args = _extract_key_args(fn_name, fn_args)
    return f"  → call {fn_name}.py ({key_args})"


def format_tool_result(fn_name: str, result: dict) -> str:
    """Format a tool result as a concise one-liner."""
    summary = _summarize_result(fn_name, result)
    return f"  ← {fn_name}.py: {summary}"


def format_engine_text(provider_name: str, content: str) -> str:
    """Format an LLM text response (truncated)."""
    clean = content.replace("\n", " ").strip()
    if len(clean) > 100:
        clean = clean[:97] + "..."
    return f'  \U0001f4ac {provider_name}: "{clean}"'


def format_agent_complete(agent_name: str, metrics: dict | None = None) -> str:
    """Format a boxed agent completion summary."""
    if metrics:
        parts = []
        for key in ["mcc", "sensitivity", "specificity", "f1"]:
            val = metrics.get(key)
            if val is not None:
                label = key.upper() if key == "mcc" else key.capitalize()[:4]
                parts.append(f"{label}={val:.3f}")
        n = metrics.get("total_evaluated", metrics.get("n", "?"))
        inner = f"{agent_name} complete (n={n}): {', '.join(parts)}"
    else:
        inner = f"{agent_name} complete"
    return f"┌ {'─' * (len(inner) + 2)} ┐\n│ {inner}   │\n└ {'─' * (len(inner) + 2)} ┘"


def format_checkpoint(title: str) -> str:
    """Format a checkpoint header."""
    return f"\n⏸  CHECKPOINT: {title}"


# ------------------------------------------------------------------
# Literature results — PubMed E-utilities lookup for citations
# ------------------------------------------------------------------

def format_literature_results(result: dict) -> str:
    """Format search_literature results as a citation list with full metadata from PubMed."""
    import xml.etree.ElementTree as ET
    try:
        import httpx as _httpx
    except ImportError:
        _httpx = None

    pmids = []
    fallback_titles = {}
    for search in result.get("searches", []):
        for r in search.get("results", []):
            url = r.get("url", "")
            title = r.get("title", "").replace(" - PubMed", "").strip()
            if "pubmed.ncbi.nlm.nih.gov/" in url:
                pmid = url.split("pubmed.ncbi.nlm.nih.gov/")[-1].split("/")[0].split("?")[0]
                if pmid and pmid.isdigit():
                    pmids.append(pmid)
                    fallback_titles[pmid] = title

    if not pmids:
        return ""

    citations = []
    if _httpx and pmids:
        try:
            resp = _httpx.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params={"db": "pubmed", "id": ",".join(pmids), "rettype": "xml"},
                timeout=10.0,
            )
            if resp.status_code == 200:
                root = ET.fromstring(resp.text)
                for article in root.findall(".//PubmedArticle"):
                    citation = _parse_pubmed_article(article)
                    if citation:
                        citations.append(citation)
        except Exception:
            pass

    fetched_pmids = {c["pmid"] for c in citations}
    for pmid in pmids:
        if pmid not in fetched_pmids:
            citations.append({
                "pmid": pmid,
                "title": fallback_titles.get(pmid, "Unknown"),
                "formatted": f"{fallback_titles.get(pmid, 'Unknown')}. PMID: {pmid}",
            })

    if citations:
        lines = [f"    • {c['formatted']}" for c in citations]
        return "\n  Papers identified:\n\n" + "\n".join(lines) + "\n"
    return ""


def _parse_pubmed_article(article) -> dict | None:
    """Parse a PubmedArticle XML element into a citation dict."""
    try:
        medline = article.find(".//MedlineCitation")
        pmid = medline.findtext("PMID", "")
        art = medline.find(".//Article")
        title = art.findtext("ArticleTitle", "").rstrip(".")

        authors = art.findall(".//Author")
        if authors:
            last = authors[0].findtext("LastName", "")
            initials = authors[0].findtext("Initials", "")
            first_author = f"{last} {initials}".strip()
            if len(authors) > 1:
                first_author += ", et al."
        else:
            first_author = ""

        journal = art.find(".//Journal")
        journal_abbr = journal.findtext("ISOAbbreviation", "") if journal is not None else ""
        pub_date = journal.find(".//PubDate") if journal is not None else None
        year = pub_date.findtext("Year", "") if pub_date is not None else ""
        if not year:
            medline_date = pub_date.findtext("MedlineDate", "") if pub_date is not None else ""
            if medline_date:
                year = medline_date[:4]

        doi = ""
        for eid in art.findall(".//ELocationID"):
            if eid.get("EIdType") == "doi":
                doi = eid.text or ""
                break
        if not doi:
            article_ids = article.findall(".//ArticleId")
            for aid in article_ids:
                if aid.get("IdType") == "doi":
                    doi = aid.text or ""
                    break

        parts = [title]
        if first_author:
            parts.append(first_author)
        journal_year = ""
        if journal_abbr:
            journal_year = journal_abbr
        if year:
            journal_year += f", {year}" if journal_year else year
        if journal_year:
            parts.append(journal_year)
        formatted = ". ".join(p.rstrip(".") for p in parts)
        if doi:
            formatted += f". DOI: {doi}"

        return {"pmid": pmid, "title": title, "formatted": formatted}
    except Exception:
        return None


# ------------------------------------------------------------------
# Argument and result summarizers per tool name
# ------------------------------------------------------------------

def _extract_key_args(fn_name: str, args: dict) -> str:
    if fn_name == "run_inference":
        ds = args.get("dataset_path", "?")
        return f'dataset="{ds}"'
    elif fn_name == "validate_results":
        pp = args.get("predictions_path", "?")
        return f'predictions="{_short_path(pp)}"'
    elif fn_name == "compute_metrics":
        pp = args.get("predictions_path", "?")
        return f'predictions="{_short_path(pp)}"'
    elif fn_name == "compare_baselines":
        mp = args.get("matrix_path") or args.get("metrics_path", "?")
        return f'matrix="{_short_path(mp)}"'
    elif fn_name == "search_literature":
        queries = args.get("queries", [])
        return f"{len(queries)} queries"
    elif fn_name == "generate_figures":
        od = args.get("output_dir", "?")
        return f'output="{_short_path(od)}"'
    elif fn_name == "write_report":
        op = args.get("output_path", "?")
        return f'output="{_short_path(op)}"'
    else:
        return ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])


def _summarize_result(fn_name: str, result: dict) -> str:
    if result.get("error"):
        return f"ERROR: {result['error'][:80]}"

    if fn_name == "run_inference":
        n = result.get("predictions_written", "?")
        errs = result.get("error_count", 0)
        return f"{n} predictions written, {errs} errors"
    elif fn_name == "validate_results":
        status = result.get("status", "?")
        nulls = result.get("null_predictions", 0)
        n = result.get("total_predictions", "?")
        if status == "pass":
            return f"PASS — {n} predictions validated, {nulls} nulls"
        else:
            issues = result.get("issues", [])
            return f"FAIL — {'; '.join(issues[:2])}"
    elif fn_name == "compute_metrics":
        s = result.get("summary", {})
        n = s.get("n", "?")
        mcc = s.get("mcc", "?")
        sens = s.get("sensitivity", "?")
        spec = s.get("specificity", "?")
        return f"n={n}, MCC={mcc}, Sens={sens}, Spec={spec}"
    elif fn_name == "compare_baselines":
        status = result.get("status", "?")
        flags = result.get("flags", [])
        summary = result.get("summary", "")
        if flags:
            return f"FLAGGED — {summary}"
        return f"PASS — {summary}"
    elif fn_name == "search_literature":
        total = result.get("total_results", 0)
        nq = result.get("total_queries", 0)
        return f"{total} papers found across {nq} queries"
    elif fn_name == "generate_figures":
        figs = result.get("figures", [])
        return f"{len(figs)} figures generated"
    elif fn_name == "write_report":
        wc = result.get("word_count", "?")
        return f"report written ({wc} words)"
    else:
        status = result.get("status", "done")
        return str(status)


def _short_path(path: str) -> str:
    """Shorten a path for display."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) > 3:
        return "/".join(parts[-3:])
    return path
