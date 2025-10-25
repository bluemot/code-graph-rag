from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from time import sleep
from typing import Any, Iterable

import mgclient
from loguru import logger


# ─────────────────────────────────────────────────────────────
# 可調整參數
# ─────────────────────────────────────────────────────────────
REL_BATCH_CHUNK = 100  # 關聯批次在單一 pattern 底下切塊大小
QUERY_MAX_RETRIES = 3  # 連線中斷/暫時錯誤時最大重試次數
RETRY_BACKOFF_SEC = 0.6  # 指數退避起始秒數


class MemgraphIngestor:
    """Handles all communication and query execution with the Memgraph database."""

    def __init__(self, host: str, port: int, batch_size: int = 1000):
        self._host = host
        self._port = port
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size
        self.conn: mgclient.Connection | None = None
        self.node_buffer: list[tuple[str, dict[str, Any]]] = []
        self.relationship_buffer: list[tuple[tuple, str, tuple, dict | None]] = []
        # 所有有可能被建立的 label 都給唯一鍵，避免 flush 被跳過
        self.unique_constraints = {
            "Project": "name",
            "Package": "qualified_name",
            "Folder": "path",
            "Module": "qualified_name",
            "Class": "qualified_name",
            "Interface": "qualified_name",
            "Enum": "qualified_name",
            "Union": "qualified_name",
            "Type": "qualified_name",
            "ModuleInterface": "qualified_name",
            "ModuleImplementation": "qualified_name",
            "Function": "qualified_name",
            "Method": "qualified_name",
            "File": "path",
            "ExternalPackage": "name",
            # CFG/Stmt
            "BasicBlock": "uid",
            "Stmt": "id",
        }

    # ─────────────────────────────────────────────────────────
    # 連線生命週期
    # ─────────────────────────────────────────────────────────
    def __enter__(self) -> "MemgraphIngestor":
        logger.info(f"Connecting to Memgraph at {self._host}:{self._port}...")
        self._connect()
        logger.info("Successfully connected to Memgraph.")
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if exc_type:
            logger.error(
                f"An exception occurred: {exc_val}. Flushing remaining items...",
                exc_info=True,
            )
        self.flush_all()
        if self.conn:
            self.conn.close()
            logger.info("\nDisconnected from Memgraph.")

    def _connect(self) -> None:
        self.conn = mgclient.connect(host=self._host, port=self._port)
        # mgclient 與 Memgraph 的 autocommit True 才會立即可見
        self.conn.autocommit = True

    # ─────────────────────────────────────────────────────────
    # 重試/自動重連
    # ─────────────────────────────────────────────────────────
    def _should_retry(self, err: Exception) -> bool:
        s = str(err).lower()
        transient_signals = [
            "failed to receive chunk size",
            "connection closed by server",
            "bad session",
            "network is down",
            "broken pipe",
            "connection reset",
            "timed out",
        ]
        return any(sig in s for sig in transient_signals)

    def _reconnect(self) -> None:
        try:
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
            logger.warning("Reconnecting to Memgraph...")
            self._connect()
            logger.info("Reconnected.")
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            raise

    def _run_with_retry(self, fn, *args, **kwargs):
        attempt = 0
        backoff = RETRY_BACKOFF_SEC
        while True:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt >= QUERY_MAX_RETRIES or not self._should_retry(e):
                    raise
                logger.warning(
                    f"[retry {attempt}/{QUERY_MAX_RETRIES}] {e}. Will reconnect and retry after {backoff:.1f}s."
                )
                self._reconnect()
                sleep(backoff)
                backoff *= 2

    # ─────────────────────────────────────────────────────────
    # 執行查詢
    # ─────────────────────────────────────────────────────────
    def _execute_query(self, query: str, params: dict[str, Any] | None = None) -> list:
        if not self.conn:
            raise ConnectionError("Not connected to Memgraph.")
        params = params or {}

        def _do():
            cursor = None
            try:
                cursor = self.conn.cursor()
                cursor.execute(query, params)
                if not cursor.description:
                    return []
                column_names = [desc.name for desc in cursor.description]
                return [dict(zip(column_names, row)) for row in cursor.fetchall()]
            finally:
                if cursor:
                    cursor.close()

        try:
            return self._run_with_retry(_do)
        except Exception as e:
            # 僅在非重複 constraint 或語法錯才輸出詳細
            if (
                "already exists" not in str(e).lower()
                and "constraint" not in str(e).lower()
            ):
                logger.error(f"!!! Cypher Error: {e}")
                logger.error(f"    Query: {query}")
                logger.error(f"    Params: {params}")
            raise

    def _execute_batch(self, query: str, params_list: list[dict[str, Any]]) -> None:
        if not self.conn or not params_list:
            return

        def _do(batch):
            cursor = None
            try:
                cursor = self.conn.cursor()
                batch_query = f"UNWIND $batch AS row\n{query}"
                cursor.execute(batch_query, {"batch": batch})
            finally:
                if cursor:
                    cursor.close()

        # 切塊送，避免單次 payload 過大
        for i in range(0, len(params_list), REL_BATCH_CHUNK):
            chunk = params_list[i : i + REL_BATCH_CHUNK]
            try:
                self._run_with_retry(_do, chunk)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.error(f"!!! Batch Cypher Error: {e}")
                    logger.error(f"    Query: {query}")
                    if len(chunk) > 10:
                        logger.error(
                            "    Params (first 10 of {}): {}...",
                            len(chunk),
                            chunk[:10],
                        )
                    else:
                        logger.error(f"    Params: {chunk}")
                raise

    # ─────────────────────────────────────────────────────────
    # 公用 API
    # ─────────────────────────────────────────────────────────
    def clean_database(self) -> None:
        logger.info("--- Cleaning database... ---")
        self._execute_query("MATCH (n) DETACH DELETE n;")
        logger.info("--- Database cleaned. ---")

    def ensure_constraints(self) -> None:
        logger.info("Ensuring constraints...")
        for label, prop in self.unique_constraints.items():
            try:
                self._execute_query(
                    f"CREATE CONSTRAINT ON (n:{label}) ASSERT n.{prop} IS UNIQUE;"
                )
            except Exception:
                # 可能已存在
                pass
        logger.info("Constraints checked/created.")

    def ensure_node_batch(self, label: str, properties: dict[str, Any]) -> None:
        """Adds a node to the buffer."""
        self.node_buffer.append((label, properties))
        if len(self.node_buffer) >= self.batch_size:
            logger.debug(
                "Node buffer reached batch size ({}). Performing incremental flush.",
                self.batch_size,
            )
            self.flush_nodes()

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, Any],
        rel_type: str,
        to_spec: tuple[str, str, Any],
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Adds a relationship to the buffer."""
        from_label, from_key, from_val = from_spec
        to_label, to_key, to_val = to_spec
        self.relationship_buffer.append(
            (
                (from_label, from_key, from_val),
                rel_type,
                (to_label, to_key, to_val),
                properties,
            )
        )
        if len(self.relationship_buffer) >= self.batch_size:
            logger.debug(
                "Relationship buffer reached batch size ({}). Performing incremental flush.",
                self.batch_size,
            )
            # 先把 node buffer 清掉，確保節點存在
            self.flush_nodes()
            self.flush_relationships()

    def flush_nodes(self) -> None:
        """Flushes the buffered nodes to the database."""
        if not self.node_buffer:
            return

        buffer_size = len(self.node_buffer)
        nodes_by_label = defaultdict(list)
        for label, props in self.node_buffer:
            nodes_by_label[label].append(props)
        flushed_total = 0
        skipped_total = 0
        for label, props_list in nodes_by_label.items():
            if not props_list:
                continue
            id_key = self.unique_constraints.get(label)
            if not id_key:
                logger.warning(
                    f"No unique constraint defined for label '{label}'. Skipping flush."
                )
                skipped_total += len(props_list)
                continue

            batch_rows: list[dict[str, Any]] = []
            for props in props_list:
                if id_key not in props:
                    logger.warning(
                        "Skipping {} node missing required '{}' property: {}",
                        label,
                        id_key,
                        props,
                    )
                    skipped_total += 1
                    continue
                row_props = {k: v for k, v in props.items() if k != id_key}
                batch_rows.append({"id": props[id_key], "props": row_props})

            if not batch_rows:
                continue

            flushed_total += len(batch_rows)
            query = f"MERGE (n:{label} {{{id_key}: row.id}})\nSET n += row.props"
            self._execute_batch(query, batch_rows)

        logger.info("Flushed {} of {} buffered nodes.", flushed_total, buffer_size)
        if skipped_total:
            logger.info(
                "Skipped {} buffered nodes due to missing identifiers or constraints.",
                skipped_total,
            )
        self.node_buffer.clear()

    def flush_relationships(self) -> None:
        if not self.relationship_buffer:
            return

        rels_by_pattern = defaultdict(list)
        for from_node, rel_type, to_node, props in self.relationship_buffer:
            pattern = (from_node[0], from_node[1], rel_type, to_node[0], to_node[1])
            rels_by_pattern[pattern].append(
                {"from_val": from_node[2], "to_val": to_node[2], "props": props or {}}
            )

        total_rel = 0
        for pattern, params_list in rels_by_pattern.items():
            from_label, from_key, rel_type, to_label, to_key = pattern
            query = (
                f"MATCH (a:{from_label} {{{from_key}: row.from_val}}), "
                f"(b:{to_label} {{{to_key}: row.to_val}})\n"
                f"MERGE (a)-[r:{rel_type}]->(b)"
            )
            if any(p["props"] for p in params_list):
                query += "\nSET r += row.props"

            self._execute_batch(query, params_list)
            total_rel += len(params_list)

        logger.info(f"Flushed {total_rel} relationships.")
        self.relationship_buffer.clear()

    def flush_all(self) -> None:
        logger.info("--- Flushing all pending writes to database... ---")
        self.flush_nodes()
        self.flush_relationships()
        logger.info("--- Flushing complete. ---")

    # 便利查詢
    def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list:
        logger.debug(f"Executing fetch query: {query} with params: {params}")
        return self._execute_query(query, params)

    def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        logger.debug(f"Executing write query: {query} with params: {params}")
        self._execute_query(query, params)

    # 匯出
    def export_graph_to_dict(self) -> dict[str, Any]:
        logger.info("Exporting graph data...")
        nodes_query = """
        MATCH (n)
        RETURN id(n) as node_id, labels(n) as labels, properties(n) as properties
        """
        nodes_data = self.fetch_all(nodes_query)

        relationships_query = """
        MATCH (a)-[r]->(b)
        RETURN id(a) as from_id, id(b) as to_id, type(r) as type, properties(r) as properties
        """
        relationships_data = self.fetch_all(relationships_query)

        graph_data = {
            "nodes": nodes_data,
            "relationships": relationships_data,
            "metadata": {
                "total_nodes": len(nodes_data),
                "total_relationships": len(relationships_data),
                "exported_at": self._get_current_timestamp(),
            },
        }

        logger.info(
            f"Exported {len(nodes_data)} nodes and {len(relationships_data)} relationships"
        )
        return graph_data

    def _get_current_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()
