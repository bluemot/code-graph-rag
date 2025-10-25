# -*- coding: utf-8 -*-
"""
Prompts configuration for the code graph RAG system — Ultra-Strict Cypher output.

Goal: completely eliminate prose from the CypherGenerator by forcing a single, deterministic query shape.
This version:
- Hard-requires output to START WITH 'MATCH ' and contain NO OTHER TEXT.
- Provides ONE canonical consolidated query template to cover "purpose of <function>" and similar lookups.
- Mirrors the same contract for both remote and local cypher models.
"""

from textwrap import dedent

GRAPH_SCHEMA_AND_RULES = dedent("""You will output ONLY a single Cypher query and NOTHING ELSE. Your output MUST start with 'MATCH ' and must not contain any commentary, code fences, or markdown.
You must not write to the DB. Only read with MATCH / OPTIONAL MATCH / RETURN / ORDER BY / LIMIT.
DISPATCHES_TO edges may originate from Module or Function. Always tolerate both qualified_name and name via case-insensitive checks.
Return aliases for all fields.
""")


# Canonical consolidated query that always works for function lookup + optional callbacks and file span.
_CONSOLIDATED_QUERY = """MATCH (f:Function)
WHERE toLower(f.name) = toLower('<TOKEN>')
   OR (f.qualified_name IS NOT NULL AND toLower(f.qualified_name) CONTAINS toLower('<TOKEN>'))
OPTIONAL MATCH (f)-[:DEFINED_IN]->(file:File)
OPTIONAL MATCH (src)-[r:DISPATCHES_TO]->(f)
RETURN f.qualified_name AS qualified_name,
       f.name AS name,
       file.path AS path,
       f.start_line AS start,
       f.end_line AS end,
       COLLECT(DISTINCT labels(src)) AS src_labels,
       COLLECT(DISTINCT coalesce(src.qualified_name, src.name)) AS sources,
       COLLECT(DISTINCT r.field) AS fields,
       COLLECT(DISTINCT r.src)   AS src_anchors,
       COLLECT(DISTINCT r.kind)  AS kinds
LIMIT 100"""


# Remote model prompt
CYPHER_SYSTEM_PROMPT = dedent(f"""You translate the user's natural-language question into ONE Cypher query ONLY.

HARD OUTPUT CONTRACT:
- Your entire output MUST be exactly one Cypher query that STARTS WITH 'MATCH '.
- Do not include explanations, comments, code fences, markdown, or any text before/after the query.
- Use the CONSOLIDATED QUERY below with <TOKEN> replaced by the best function token from the user question.
- Keep the query shape unchanged. Do not add clauses. Do not remove aliases. Do not change RETURN columns.

CONSOLIDATED QUERY (DO NOT ALTER SHAPE; ONLY REPLACE <TOKEN>):
{_CONSOLIDATED_QUERY}
""")


# Local model prompt
LOCAL_CYPHER_SYSTEM_PROMPT = CYPHER_SYSTEM_PROMPT


# Orchestrator: unchanged, but reiterate the graph-first/files-second rule
RAG_ORCHESTRATOR_SYSTEM_PROMPT = dedent("""Graph-first, Files-second.

1) Run the Cypher from the Cypher system prompt to retrieve function nodes, optional DEFINED_IN, and incoming DISPATCHES_TO edges.
2) If DEFINED_IN is missing but file/span is needed, open likely files by module path or name match to extract the span.
3) When callbacks are involved, surface edge properties {field, src, kind} in the final answer.
4) Only read operations.
""")


__all__ = [
    "GRAPH_SCHEMA_AND_RULES",
    "CYPHER_SYSTEM_PROMPT",
    "LOCAL_CYPHER_SYSTEM_PROMPT",
    "RAG_ORCHESTRATOR_SYSTEM_PROMPT",
]
