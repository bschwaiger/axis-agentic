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
    write_report as _write_report_original,
    _load_json,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------
# write_report (wrapper with better error handling)
# ------------------------------------------------------------------

def write_report(
    matrix_path: str,
    comparison_path: str,
    literature_results: str,
    figures_dir: str,
    output_path: str,
) -> dict:
    """Wrapper around original write_report with fallback for missing literature."""
    # If literature_results path doesn't exist, try to find it or create empty
    lit_path = Path(literature_results)
    if not lit_path.is_absolute():
        lit_path = PROJECT_ROOT / lit_path
    if not lit_path.exists():
        # Try to find it in the same directory as the matrix
        matrix_dir = Path(matrix_path)
        if not matrix_dir.is_absolute():
            matrix_dir = PROJECT_ROOT / matrix_dir
        alt_path = matrix_dir.parent / "literature_results.json"
        if alt_path.exists():
            literature_results = str(alt_path)
        else:
            # Create empty literature results so report generation doesn't crash
            alt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(alt_path, "w") as f:
                json.dump({"status": "not_found", "searches": []}, f)
            literature_results = str(alt_path)

    return _write_report_original(
        matrix_path=matrix_path,
        comparison_path=comparison_path,
        literature_results=literature_results,
        figures_dir=figures_dir,
        output_path=output_path,
    )


# ------------------------------------------------------------------
# search_literature (Anthropic web search replacement)
# ------------------------------------------------------------------

def search_literature(queries: list[str], max_results_per_query: int = 3, output_dir: str = "") -> dict:
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
                    "allowed_domains": ["pubmed.ncbi.nlm.nih.gov"],
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search PubMed for: {query}\n\n"
                        f"Return up to {max_results_per_query} relevant papers from PubMed. "
                        f"For each paper, provide: title, PubMed URL, and a one-sentence summary."
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

    # Determine output path: prefer explicit output_dir, then find from matrix
    from datetime import datetime as _dt
    import glob
    if output_dir:
        out_d = Path(output_dir)
        if not out_d.is_absolute():
            out_d = PROJECT_ROOT / out_d
        out = out_d / "literature_results.json"
    else:
        # Fallback: find the most recent comparison_matrix.json
        matrices = sorted(glob.glob(str(PROJECT_ROOT / "results" / "*" / "comparison_matrix.json")))
        if matrices:
            out = Path(matrices[-1]).parent / "literature_results.json"
        else:
            out = PROJECT_ROOT / "results" / f"literature_{_dt.now().strftime('%y%m%d%H%M')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    result["literature_path"] = str(out)
    # Also return the path explicitly so Claude can pass it to write_report
    result["output_note"] = f"Literature results written to {out}. Pass this path as literature_results to write_report."
    return result


def _parse_search_response(response) -> list[dict]:
    """Parse an Anthropic web search response into structured paper results.

    Response content blocks follow this pattern:
    1. text — Claude's decision to search
    2. server_tool_use — The search query
    3. web_search_tool_result — Array of search result objects with url, title, page_age, encrypted_content
    4. text — Claude's summary with citations

    We extract from web_search_tool_result blocks (structured) and fall back
    to parsing URLs from text blocks.
    """
    hits = []
    seen_urls = set()

    # First pass: extract from web_search_tool_result blocks
    for block in response.content:
        block_type = getattr(block, 'type', '')
        if block_type == "web_search_tool_result":
            # The block contains a 'content' list of search result objects
            content_list = getattr(block, 'content', [])
            if not isinstance(content_list, list):
                content_list = []
            for item in content_list:
                item_type = getattr(item, 'type', '')
                if item_type == 'web_search_result':
                    url = getattr(item, 'url', '')
                    title = getattr(item, 'title', '')
                    # encrypted_content is not human-readable; page_age is metadata
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        hits.append({
                            "title": title.replace(" - PubMed", "").strip(),
                            "url": url,
                            "content": "",  # no plaintext snippet available from web search API
                        })

    # Second pass: extract cited text from Claude's text blocks (has citations)
    for block in response.content:
        if getattr(block, 'type', '') == "text" and hasattr(block, 'text') and block.text:
            # Try to enrich existing hits with context from the text
            import re
            # Also catch any PubMed URLs not in structured results
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
