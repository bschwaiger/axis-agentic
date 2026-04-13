"""Tool implementations for the Analyst agent (Anthropic variant).

Only search_literature changes: Tavily -> Anthropic web search (web_search_20250305).
All other tools (compare_baselines, generate_figures, write_report) are imported
unchanged from the original analyst_impl.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Re-use unchanged tools from original implementation
from tools.analyst_impl import (
    compare_baselines,
    generate_figures,
    write_report,
    _load_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# search_literature (Anthropic web search replacement)
# ------------------------------------------------------------------

def search_literature(queries: list[str], max_results_per_query: int = 3) -> dict:
    """Search PubMed via Anthropic web search tool.

    Uses Claude with the web_search_20250305 server tool to search PubMed.
    Each query is scoped to PubMed by prepending "site:pubmed.ncbi.nlm.nih.gov"
    to the search instruction.
    """
    import anthropic
    from cockpit import events as cockpit_events
    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = anthropic.Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def _search_one(qi: int, query: str) -> dict:
        cockpit_events.emit("web_search", query=query, query_index=qi + 1,
                            total_queries=len(queries), domain="pubmed.ncbi.nlm.nih.gov")
        try:
            t0 = _time.monotonic()
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                tools=[{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": max_results_per_query + 2,
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search PubMed for: {query}\n\n"
                        f"Focus on results from pubmed.ncbi.nlm.nih.gov. "
                        f"Return up to {max_results_per_query} relevant papers. "
                        f"For each paper, provide: title, PubMed URL, authors (first author et al.), "
                        f"journal, year, and a one-sentence summary of the findings."
                    ),
                }],
            )
            elapsed_ms = int((_time.monotonic() - t0) * 1000)

            # Parse results from Claude's response
            hits = _parse_search_response(response)
            cockpit_events.emit("web_search_result", query=query, hits=len(hits),
                                elapsed_ms=elapsed_ms)
            return {"query": query, "results": hits[:max_results_per_query], "result_count": len(hits[:max_results_per_query])}
        except Exception as e:
            cockpit_events.emit("web_search_result", query=query, hits=0, error=str(e))
            return {"query": query, "results": [], "result_count": 0, "error": str(e)}

    # Run queries concurrently (typically 2-4 queries)
    all_results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
        futures = {executor.submit(_search_one, qi, q): qi for qi, q in enumerate(queries)}
        for future in as_completed(futures):
            idx = futures[future]
            all_results[idx] = future.result()

    result = {
        "status": "success",
        "total_queries": len(queries),
        "total_results": sum(r["result_count"] for r in all_results),
        "searches": all_results,
    }

    # Write next to the comparison matrix if one exists
    from datetime import datetime as _dt
    fallback = PROJECT_ROOT / "results" / f"literature_{_dt.now().strftime('%y%m%d%H%M')}.json"
    import glob
    matrices = sorted(glob.glob(str(PROJECT_ROOT / "results" / "*" / "comparison_matrix.json")))
    if matrices:
        out = Path(matrices[-1]).parent / "literature_results.json"
    else:
        out = fallback
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    result["literature_path"] = str(out)
    return result


def _parse_search_response(response) -> list[dict]:
    """Parse an Anthropic web search response into structured paper results.

    Extracts information from both web_search_tool_result blocks (which contain
    the actual search results with URLs) and text blocks (which contain Claude's
    formatted summary).
    """
    hits = []
    seen_urls = set()

    # First pass: extract structured results from web_search_tool_result blocks
    for block in response.content:
        if hasattr(block, 'type') and block.type == "web_search_tool_result":
            for search_result in getattr(block, 'search_results', []):
                url = getattr(search_result, 'url', '')
                title = getattr(search_result, 'title', '')
                snippet = getattr(search_result, 'snippet', '')
                # Only include PubMed results
                if 'pubmed.ncbi.nlm.nih.gov' in url and url not in seen_urls:
                    seen_urls.add(url)
                    hits.append({
                        "title": title.replace(" - PubMed", "").strip(),
                        "url": url,
                        "content": snippet[:500] if snippet else "",
                    })

    # Second pass: if no structured results, parse from text
    if not hits:
        for block in response.content:
            if hasattr(block, 'type') and block.type == "text" and block.text:
                # Try to extract PubMed URLs and titles from text
                import re
                # Match URLs
                urls = re.findall(r'https?://pubmed\.ncbi\.nlm\.nih\.gov/\d+/?', block.text)
                for url in urls:
                    if url not in seen_urls:
                        seen_urls.add(url)
                        hits.append({
                            "title": "",
                            "url": url,
                            "content": "",
                        })

    return hits


# ------------------------------------------------------------------
# Dispatch
# ------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "compare_baselines": compare_baselines,
    "search_literature": search_literature,
    "generate_figures": generate_figures,
    "write_report": write_report,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name, return JSON string result."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(**arguments)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})
