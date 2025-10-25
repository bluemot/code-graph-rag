# llm.py — Generic Flow (Graph→Qdrant→Code) + Single-Function Context
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic_ai import Agent, Tool

from ..config import settings
from ..prompts import (
    CYPHER_SYSTEM_PROMPT,
    LOCAL_CYPHER_SYSTEM_PROMPT,
    RAG_ORCHESTRATOR_SYSTEM_PROMPT,
)
from ..providers.base import get_provider
from ..services.graph_service import MemgraphIngestor  # ← Graph 連線

# ---- Optional deps for Qdrant + embeddings（可缺省，缺就不啟用 Qdrant 工具） ----
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Filter,
        FieldCondition,
        MatchValue,
        MatchText,
        Range,
    )
except Exception:
    QdrantClient = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import requests
except Exception:
    requests = None  # type: ignore


class LLMGenerationError(Exception):
    """Custom exception for LLM generation failures."""
    pass


def _clean_cypher_response(response_text: str) -> str:
    """Utility to clean up common LLM formatting artifacts from a Cypher query."""
    query = response_text.strip().replace("`", "")
    if query.startswith("cypher"):
        query = query[6:].strip()
    if not query.endswith(";"):
        query += ";"
    return query


class CypherGenerator:
    """Generates Cypher queries from natural language."""

    def __init__(self) -> None:
        try:
            config = settings.active_cypher_config
            provider = get_provider(
                config.provider,
                api_key=config.api_key,
                endpoint=config.endpoint,
                project_id=config.project_id,
                region=config.region,
                provider_type=config.provider_type,
                thinking_budget=config.thinking_budget,
            )
            llm = provider.create_model(config.model_id)
            system_prompt = (
                LOCAL_CYPHER_SYSTEM_PROMPT
                if config.provider == "ollama"
                else CYPHER_SYSTEM_PROMPT
            )
            self.agent = Agent(
                model=llm,
                system_prompt=system_prompt,
                output_type=str,
            )
        except Exception as e:
            raise LLMGenerationError(
                f"Failed to initialize CypherGenerator: {e}"
            ) from e

    async def generate(self, natural_language_query: str) -> str:
        logger.info(
            f"  [CypherGenerator] Generating query for: '{natural_language_query}'"
        )
        try:
            result = await self.agent.run(natural_language_query)
            if (
                not isinstance(result.output, str)
                or "MATCH" not in result.output.upper()
            ):
                raise LLMGenerationError(
                    f"LLM did not generate a valid query. Output: {result.output}"
                )
            query = _clean_cypher_response(result.output)
            logger.info(f"  [CypherGenerator] Generated Cypher: {query}")
            return query
        except Exception as e:
            logger.error(f"  [CypherGenerator] Error: {e}")
            raise LLMGenerationError(f"Cypher generation failed: {e}") from e


# =========================
# Qdrant 檢索與工具封裝
# =========================

class _Embedder:
    """最小化的查詢嵌入器，與上傳時保持一致（.env 控制）。"""
    def __init__(self) -> None:
        self.backend = (os.getenv("EMBEDDING_BACKEND", "SENTENCE_TRANSFORMERS") or "SENTENCE_TRANSFORMERS").upper()
        self.model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        if self.backend == "SENTENCE_TRANSFORMERS":
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed")
            device = os.getenv("EMBEDDING_DEVICE", None)  # e.g. 'cpu' or 'cuda'
            self._st = SentenceTransformer(self.model, device=device)
        elif self.backend == "OLLAMA":
            if requests is None:
                raise RuntimeError("requests is required for OLLAMA backend")
            self._ollama = os.getenv("OLLAMA_ENDPOINT", str(settings.EMBEDDING_ENDPOINT))
            if not self._ollama:
                raise RuntimeError("OLLAMA_ENDPOINT not set")
        else:
            raise RuntimeError(f"Unknown EMBEDDING_BACKEND: {self.backend}")

    def embed_one(self, text: str) -> List[float]:
        if self.backend == "SENTENCE_TRANSFORMERS":
            return self._st.encode([text], normalize_embeddings=True)[0].tolist()
        if self.backend == "OLLAMA":
            r = requests.post(
                f"{self._ollama}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=120,
            )
            r.raise_for_status()
            return r.json()["embedding"]  # type: ignore
        raise AssertionError("unreachable")


def _connect_qdrant() -> Optional[QdrantClient]:
    if QdrantClient is None:
        logger.warning("qdrant-client not installed; Qdrant tools disabled.")
        return None
    url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    api_key = os.getenv("QDRANT_API_KEY", None)
    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    timeout = float(os.getenv("QDRANT_TIMEOUT", "60"))
    try:
        return QdrantClient(url=url, api_key=api_key or None, prefer_grpc=prefer_grpc, timeout=timeout)
    except Exception as e:
        logger.warning(f"Failed to connect Qdrant: {e}; Qdrant tools disabled.")
        return None


# ---------- 通用前置工具：抽識別字 / 系統流程 ----------
def _extract_identifiers(question: str) -> list[str]:
    """從問題中抓出可能的 C/driver 識別字（函式名、結構等）。"""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", question or "")
    seen, out = set(), []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(tl)
    return out[:10]


def _build_prepare_code_context_tool() -> Optional[Tool]:
    """
    單函式/局部問題：抽識別字→(Graph)找 (qn, path, start,end) → (Qdrant) 取 code。
    """
    client = _connect_qdrant()
    if client is None:
        return None
    collection = os.getenv("QDRANT_COLLECTION", "code_chunks")

    def _get_snippets_by_qn(qn: str) -> List[Tuple[int, int, str]]:
        all_points, next_page = [], None
        while True:
            points, next_page = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=[FieldCondition(key="qualified_name", match=MatchValue(value=qn))]),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_page,
            )
            all_points.extend(points)
            if next_page is None:
                break
        chunks: List[Tuple[int, int, str]] = []
        for p in all_points:
            pl = p.payload or {}
            chunks.append((int(pl.get("start", 0)), int(pl.get("end", 0)), pl.get("code", "")))
        chunks.sort(key=lambda x: x[0])
        return chunks

    def prepare_code_context(question: str) -> dict:
        idents = _extract_identifiers(question)
        if not idents:
            return {"candidates": []}

        query = """
        MATCH (f:Function)-[:DEFINED_IN]->(file:File)
        WHERE toLower(f.name) IN $names
           OR any(n IN $names WHERE toLower(f.qualified_name) CONTAINS n)
        RETURN f.qualified_name AS qn, file.path AS path, f.start_line AS start, f.end_line AS end
        LIMIT 30
        """
        params = {"names": idents}
        logger.debug("[prepare_code_context] Cypher:\n%s\nparams=%s", query, params)

        try:
            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=1000,
            ) as ing:
                rows = ing.fetch_all(query, params) or []
        except Exception as e:
            return {"error": f"memgraph_error: {e}", "candidates": []}

        results: List[Dict[str, Any]] = []
        seen = set()
        for r in rows:
            qn = r.get("qn")
            path = r.get("path")
            start = int(r.get("start") or 0)
            end = int(r.get("end") or 0)
            if not qn or not path or not start or not end:
                continue
            key = (qn, path)
            if key in seen:
                continue
            seen.add(key)
            try:
                chunks = _get_snippets_by_qn(qn)
            except Exception:
                chunks = []
            results.append(
                {
                    "qualified_name": qn,
                    "path": path,
                    "start": start,
                    "end": end,
                    "snippets": [{"start": s, "end": e, "code": c} for (s, e, c) in chunks],
                }
            )
            if len(results) >= 10:
                break

        return {"candidates": results}

    return Tool(
        prepare_code_context,
        name="prepare_code_context",
        description=(
            "Prepare localized code context from user's question: "
            "extract probable identifiers, query graph for (qualified_name, path, start, end), "
            "then fetch ACTUAL code snippets from Qdrant. "
            "Return JSON with 'candidates' listing functions and their snippets."
        ),
    )


def _build_prepare_flow_context_tool() -> Optional[Tool]:
    """
    系統層級/跨層級：以 kind 關鍵詞（不綁特定詞）在圖上找流程相關節點，並為每個節點取回 Qdrant 片段。
    """
    client = _connect_qdrant()
    if client is None:
        return None
    collection = os.getenv("QDRANT_COLLECTION", "code_chunks")

    def _get_snippets_by_qn(qn: str):
        all_points, next_page = [], None
        while True:
            points, next_page = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(must=[FieldCondition(key="qualified_name", match=MatchValue(value=qn))]),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_page,
            )
            all_points.extend(points)
            if next_page is None:
                break
        chunks = []
        for p in all_points:
            pl = p.payload or {}
            chunks.append({"start": int(pl.get("start", 0)), "end": int(pl.get("end", 0)), "code": pl.get("code", "")})
        chunks.sort(key=lambda x: x["start"])
        return chunks

    def prepare_flow_context(kind: str) -> dict:
        kw = (kind or "").strip().lower()
        if not kw:
            return {"kind": kw, "nodes": []}
    
        # --- 1) DISPATCHES_TO + 名稱/QN 正規表達式（詞元級），並過濾負關鍵字 ---
        # 只收：r.field 含 scan，或 函式名/QN 的 token 恰好是 scan
        # 排除：deny/interval/thr/sched/abort/stop/debug/gscan
        q1 = """
        MATCH (m:Module)-[r:DISPATCHES_TO]->(f:Function)-[:DEFINED_IN]->(file:File)
        WHERE (
            toLower(r.field) CONTAINS $kw
            OR toLower(f.name) =~ '(^|.*[_\\W])' + $kw + '([_\\W].*|$)'
            OR (f.qualified_name IS NOT NULL AND toLower(f.qualified_name) =~ '(^|.*[_.\\W])' + $kw + '([_.\\W].*|$)')
        )
        AND NOT toLower(f.name) =~ '.*(deny|interval|thr|sched|abort|stop|debug|gscan).*'
        AND NOT (f.qualified_name IS NOT NULL AND toLower(f.qualified_name) =~ '.*(deny|interval|thr|sched|abort|stop|debug|gscan).*')
        RETURN m.qualified_name AS m_qn, f.qualified_name AS qn, file.path AS path,
               f.start_line AS start, f.end_line AS end, r.field AS field, r.kind AS edge_kind
        LIMIT 300
        """
    
        # --- 2) 名稱/ QN 另一輪補強（同樣詞元級 + 負關鍵字） ---
        q2 = """
        MATCH (f:Function)-[:DEFINED_IN]->(file:File)
        WHERE (
            toLower(f.name) =~ '(^|.*[_\\W])' + $kw + '([_\\W].*|$)'
            OR (f.qualified_name IS NOT NULL AND toLower(f.qualified_name) =~ '(^|.*[_.\\W])' + $kw + '([_.\\W].*|$)')
        )
        AND NOT toLower(f.name) =~ '.*(deny|interval|thr|sched|abort|stop|debug|gscan).*'
        AND NOT (f.qualified_name IS NOT NULL AND toLower(f.qualified_name) =~ '.*(deny|interval|thr|sched|abort|stop|debug|gscan).*')
        RETURN null AS m_qn, f.qualified_name AS qn, file.path AS path, f.start_line AS start, f.end_line AS end,
               null AS field, null AS edge_kind
        LIMIT 600
        """
    
        # --- 3) CALLS 一跳（保持原本邏輯即可） ---
        q3 = """
        MATCH (src:Function)-[:DEFINED_IN]->(file1:File)
        WHERE (
            toLower(src.name) =~ '(^|.*[_\\W])' + $kw + '([_\\W].*|$)'
            OR (src.qualified_name IS NOT NULL AND toLower(src.qualified_name) =~ '(^|.*[_.\\W])' + $kw + '([_.\\W].*|$)')
        )
        OPTIONAL MATCH (src)-[:CALLS]->(callee:Function)-[:DEFINED_IN]->(file2:File)
        RETURN src.qualified_name AS src_qn, file1.path AS src_path, src.start_line AS src_start, src.end_line AS src_end,
               callee.qualified_name AS callee_qn, file2.path AS callee_path, callee.start_line AS callee_start, callee.end_line AS callee_end
        LIMIT 600
        """
    
        params = {"kw": kw}
        logger.debug("[prepare_flow_context] Cypher q1/q2/q3 with params=%s", params)
    
        def _layer_of(path: str, qn: Optional[str]) -> str:
            p = (path or "").lower()
            q = (qn or "").lower()
            if "ioctl_cfg80211" in p or "cfg80211" in q:
                return "cfg80211"
            if "/phl/" in p or ".phl." in q:
                return "phl"
            if "/hal" in p or "/phy/" in p or "/rf/" in p:
                return "hal"
            return "other"
    
        # 收集節點
        nodes: dict[tuple, dict] = {}
        try:
            with MemgraphIngestor(host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT, batch_size=1000) as ing:
                for r in ing.fetch_all(q1, params) or []:
                    key = (r["qn"], r["path"])
                    if not r["qn"] or not r["path"] or not r["start"] or not r["end"]:
                        continue
                    nodes.setdefault(key, {
                        "qualified_name": r["qn"], "path": r["path"],
                        "start": int(r["start"]), "end": int(r["end"]),
                        "layer": _layer_of(r["path"], r["qn"]),
                        "via": "DISPATCHES_TO", "field": r.get("field"), "kind": r.get("edge_kind"),
                        "snippets": []
                    })
    
                for r in ing.fetch_all(q2, params) or []:
                    key = (r["qn"], r["path"])
                    if not r["qn"] or not r["path"] or not r["start"] or not r["end"]:
                        continue
                    nodes.setdefault(key, {
                        "qualified_name": r["qn"], "path": r["path"],
                        "start": int(r["start"]), "end": int(r["end"]),
                        "layer": _layer_of(r["path"], r["qn"]),
                        "via": "NAME_MATCH", "field": None, "kind": None,
                        "snippets": []
                    })
    
                try:
                    for r in ing.fetch_all(q3, params) or []:
                        if r.get("src_qn") and r.get("src_path") and r.get("src_start") and r.get("src_end"):
                            key = (r["src_qn"], r["src_path"])
                            nodes.setdefault(key, {
                                "qualified_name": r["src_qn"], "path": r["src_path"],
                                "start": int(r["src_start"]), "end": int(r["src_end"]),
                                "layer": _layer_of(r["src_path"], r["src_qn"]),
                                "via": "CALLS", "field": None, "kind": None,
                                "snippets": []
                            })
                        if r.get("callee_qn") and r.get("callee_path") and r.get("callee_start") and r.get("callee_end"):
                            key = (r["callee_qn"], r["callee_path"])
                            nodes.setdefault(key, {
                                "qualified_name": r["callee_qn"], "path": r["callee_path"],
                                "start": int(r["callee_start"]), "end": int(r["callee_end"]),
                                "layer": _layer_of(r["callee_path"], r["callee_qn"]),
                                "via": "CALLS", "field": None, "kind": None,
                                "snippets": []
                            })
                except Exception:
                    pass
        except Exception as e:
            return {"error": f"memgraph_error: {e}", "nodes": []}
    
        # ---- 補：從 Qdrant 抓 snippets（沿用你原本的 _get_snippets_by_qn） ----
        for key, n in list(nodes.items()):
            try:
                n["snippets"] = _get_snippets_by_qn(n["qualified_name"])
            except Exception:
                n["snippets"] = []
    
        # ---- 新增：後處理排序打分，壓低 scan_deny / sched_scan 等 ----
        NEG = ("deny", "interval", "thr", "sched", "abort", "stop", "debug", "gscan")
        def score(n: dict) -> float:
            name = (n["qualified_name"] or "").lower()
            field = (n.get("field") or "").lower()
            layer_order = {"cfg80211": 0, "phl": 1, "hal": 2, "other": 3}
            s = 0.0
            # 來源權重
            if n["via"] == "DISPATCHES_TO": s += 5
            elif n["via"] == "CALLS":       s += 2
            else:                           s += 1
            # field 精準命中
            if field == kw: s += 4
            # 名稱 token 為 scan（避免 scan_deny）
            if re.search(r'(^|[_.])' + re.escape(kw) + r'([_.]|$)', name): s += 3
            # 負關鍵字扣分
            if any(k in name for k in NEG): s -= 5
            # 層級順序
            s += max(0, 3 - layer_order.get(n["layer"], 3)) * 0.5  # cfg80211 最前
            return s
    
        ordered = sorted(nodes.values(), key=score, reverse=True)
    
        return {"kind": kw, "nodes": ordered}


def _build_qdrant_tools() -> List[Tool]:
    """
    建立工具：
      - prepare_flow_context(kind)  [System-level 通用]
      - prepare_code_context(question) [單函式/局部]
      - qdrant_semantic_search(query_text, path_contains=None, top_k=5)
      - qdrant_get_function_snippets_by_qn(qn)
      - qdrant_get_snippets_by_path_range(path, start, end)
    """
    client = _connect_qdrant()
    if client is None:
        return []

    collection = os.getenv("QDRANT_COLLECTION", "code_chunks")

    # 嵌入器：只在 semantic_search 用到
    def _get_embedder() -> _Embedder:
        return _Embedder()

    tools: List[Tool] = []

    # 先加入兩個「通用前置工具」
    flow_tool = _build_prepare_flow_context_tool()
    if flow_tool:
        tools.append(flow_tool)

    code_tool = _build_prepare_code_context_tool()
    if code_tool:
        tools.append(code_tool)

    # --- Tool: semantic search ---
    def qdrant_semantic_search(
        query_text: str,
        path_contains: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        embedder = _get_embedder()
        vec = embedder.embed_one(query_text)

        query_filter = None
        if path_contains:
            try:
                query_filter = Filter(must=[FieldCondition(key="path", match=MatchText(text=path_contains))])
            except Exception:
                query_filter = Filter(must=[FieldCondition(key="path", match={"text": path_contains})])  # type: ignore

        try:
            result = client.query_points(
                collection_name=collection,
                query=vec,
                limit=max(1, int(top_k)),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            hits = result.points
        except Exception:
            result = client.search(
                collection_name=collection,
                query_vector=vec,
                limit=max(1, int(top_k)),
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            hits = result

        out: List[Dict[str, Any]] = []
        for h in hits:
            p = h.payload or {}
            out.append(
                {
                    "qualified_name": p.get("qualified_name"),
                    "path": p.get("path"),
                    "start": p.get("start"),
                    "end": p.get("end"),
                    "language": p.get("language"),
                    "score": getattr(h, "score", None),
                    "code_head": "\n".join((p.get("code") or "").splitlines()[:8]),
                }
            )
        return out

    # --- Tool: get function snippets by qualified_name ---
    def qdrant_get_function_snippets_by_qn(qn: str) -> List[Tuple[int, int, str]]:
        all_points = []
        next_page = None
        while True:
            points, next_page = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="qualified_name", match=MatchValue(value=qn))]
                ),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_page,
            )
            all_points.extend(points)
            if next_page is None:
                break

        chunks: List[Tuple[int, int, str]] = []
        for p in all_points:
            pl = p.payload or {}
            chunks.append((int(pl.get("start", 0)), int(pl.get("end", 0)), pl.get("code", "")))
        chunks.sort(key=lambda x: x[0])
        return chunks

    # --- Tool: get snippets by path range ---
    def qdrant_get_snippets_by_path_range(
        path: str,
        fn_start: int,
        fn_end: int,
    ) -> List[Tuple[int, int, str]]:
        all_points = []
        next_page = None
        while True:
            points, next_page = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="path", match=MatchValue(value=path)),
                        FieldCondition(key="start", range=Range(lte=int(fn_end))),
                        FieldCondition(key="end", range=Range(gte=int(fn_start))),
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=256,
                offset=next_page,
            )
            all_points.extend(points)
            if next_page is None:
                break

        chunks: List[Tuple[int, int, str]] = []
        for p in all_points:
            pl = p.payload or {}
            chunks.append((int(pl.get("start", 0)), int(pl.get("end", 0)), pl.get("code", "")))
        chunks.sort(key=lambda x: x[0])
        return chunks

    tools.extend(
        [
            Tool(
                qdrant_semantic_search,
                name="qdrant_semantic_search",
                description=(
                    "Semantic search over code chunks in Qdrant. "
                    "Args: query_text: str, path_contains: Optional[str]=None, top_k: int=5. "
                    "Returns a list of hits with qualified_name, path, start, end, score, code_head."
                ),
            ),
            Tool(
                qdrant_get_function_snippets_by_qn,
                name="qdrant_get_function_snippets_by_qn",
                description=(
                    "Get all code snippets for a given function qualified_name from Qdrant, "
                    "sorted by start line. Returns List[Tuple[start, end, code]]."
                ),
            ),
            Tool(
                qdrant_get_snippets_by_path_range,
                name="qdrant_get_snippets_by_path_range",
                description=(
                    "Get code snippets overlapping a [start,end] range in a file path from Qdrant. "
                    "Useful when Memgraph returns file path and function line span. "
                    "Returns List[Tuple[start, end, code]]."
                ),
            ),
        ]
    )
    return tools


def _augment_system_prompt(sp: str) -> str:
    """
    在原本的 RAG 系統提示前，追加強制流程與使用指令，避免 LLM 只靠猜測。
    """
    prefix = """
You are a codebase RAG assistant. STRICT WORKFLOW:

- For SYSTEM-LEVEL or CROSS-LAYER questions (e.g., "how does the driver complete the whole <X>?", "整個<X>流程怎麼跑"):
  A) MUST call prepare_flow_context("<X>") FIRST. (generic 'kind'; not hard-coded)
  B) If you still need exact code for a specific function, call prepare_code_context(<original question>)
     or qdrant_get_function_snippets_by_qn for that qualified_name.
  C) Explain the flow step-by-step, grounding each step with code snippets and cite (path:start-end).
     DO NOT speculate beyond retrieved code.

- For single-function questions:
  A) MUST call prepare_code_context(<question>) FIRST, then answer strictly from returned snippets.

- General rules:
  • Use Qdrant tools to fetch ACTUAL source code before answering.
  • Prefer qdrant_get_function_snippets_by_qn(qualified_name); if only path + [start,end] is known, use qdrant_get_snippets_by_path_range.
  • If you couldn't retrieve code context, say so explicitly and avoid speculation.
""".strip()
    return prefix + "\n\n" + sp


def create_rag_orchestrator(tools: list[Tool]) -> Agent:
    """Factory function to create the main RAG orchestrator agent."""
    try:
        config = settings.active_orchestrator_config
        provider = get_provider(
            config.provider,
            api_key=config.api_key,
            endpoint=config.endpoint,
            project_id=config.project_id,
            region=config.region,
            provider_type=config.provider_type,
            thinking_budget=config.thinking_budget,
        )
        llm = provider.create_model(config.model_id)

        # 追加 Qdrant 工具（若環境允許）
        qdrant_tools = _build_qdrant_tools()
        if qdrant_tools:
            logger.info(f"Enabled {len(qdrant_tools)} Qdrant tools for RAG orchestration.")
            tools = tools + qdrant_tools
        else:
            logger.warning("Qdrant tools are not enabled (missing deps or env).")

        # 強化系統提示：System-level → prepare_flow_context；Single-function → prepare_code_context
        system_prompt = _augment_system_prompt(RAG_ORCHESTRATOR_SYSTEM_PROMPT)

        return Agent(
            model=llm,
            system_prompt=system_prompt,
            tools=tools,
        )
    except Exception as e:
        raise LLMGenerationError(f"Failed to initialize RAG Orchestrator: {e}") from e
