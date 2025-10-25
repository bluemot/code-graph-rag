"""Structure processor for identifying packages and folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from ..config import IGNORE_PATTERNS
from ..services.graph_service import MemgraphIngestor


class StructureProcessor:
    """Handles identification and processing of project structure."""

    def __init__(
        self,
        ingestor: MemgraphIngestor,
        repo_path: Path,
        project_name: str,
        queries: dict[str, Any],
    ):
        self.ingestor = ingestor
        self.repo_path = Path(repo_path)
        self.project_name = project_name
        self.queries = queries
        # key: 相對於 repo_root 的資料夾 Path；value: 若為套件則是其 qualified_name，否則 None（一般資料夾）
        self.structural_elements: dict[Path, str | None] = {}
        self.ignore_dirs = IGNORE_PATTERNS

    # ─────────────────────────────────────────────────────────
    # Pass 1: 掃描資料夾 → 建立 Package / Folder 階層
    # ─────────────────────────────────────────────────────────
    def identify_structure(self) -> None:
        """First pass: Efficiently walks the directory to find all packages and folders."""

        def should_skip_dir(path: Path) -> bool:
            """Check if directory should be skipped based on ignore patterns."""
            return any(part in self.ignore_dirs for part in path.parts)

        # 收集所有資料夾（含 root）
        directories = {self.repo_path}
        for path in self.repo_path.rglob("*"):
            if path.is_dir() and not should_skip_dir(path.relative_to(self.repo_path)):
                directories.add(path)

        # 以穩定順序處理
        for root in sorted(directories):
            relative_root = root.relative_to(self.repo_path)

            parent_rel_path = relative_root.parent
            parent_container_qn = self.structural_elements.get(parent_rel_path)

            # 是否為某語言的「套件」資料夾（例如 Python 的 __init__.py 等）
            is_package = False
            package_indicators = set()

            # 從所有語言的 config 聚合 package 指示檔名
            for _, lang_queries in self.queries.items():
                lang_config = lang_queries["config"]
                package_indicators.update(lang_config.package_indicators)

            # 有任一指示檔即視為套件
            for indicator in package_indicators:
                if (root / indicator).exists():
                    is_package = True
                    break

            if is_package:
                package_qn = ".".join([self.project_name] + list(relative_root.parts))
                self.structural_elements[relative_root] = package_qn
                logger.info(f"  Identified Package: {package_qn}")

                self.ingestor.ensure_node_batch(
                    "Package",
                    {
                        "qualified_name": package_qn,
                        "name": root.name,
                        "path": str(relative_root),  # Folder 相對路徑
                    },
                )

                parent_label, parent_key, parent_val = (
                    ("Project", "name", self.project_name)
                    if parent_rel_path == Path(".")
                    else (
                        ("Package", "qualified_name", parent_container_qn)
                        if parent_container_qn
                        else ("Folder", "path", str(parent_rel_path))
                    )
                )
                self.ingestor.ensure_relationship_batch(
                    (parent_label, parent_key, parent_val),
                    "CONTAINS_PACKAGE",
                    ("Package", "qualified_name", package_qn),
                )

            elif root != self.repo_path:
                # 一般資料夾
                self.structural_elements[relative_root] = None
                logger.info(f"  Identified Folder: '{relative_root}'")

                self.ingestor.ensure_node_batch(
                    "Folder", {"path": str(relative_root), "name": root.name}
                )

                parent_label, parent_key, parent_val = (
                    ("Project", "name", self.project_name)
                    if parent_rel_path == Path(".")
                    else (
                        ("Package", "qualified_name", parent_container_qn)
                        if parent_container_qn
                        else ("Folder", "path", str(parent_rel_path))
                    )
                )
                self.ingestor.ensure_relationship_batch(
                    (parent_label, parent_key, parent_val),
                    "CONTAINS_FOLDER",
                    ("Folder", "path", str(relative_root)),
                )

    # ─────────────────────────────────────────────────────────
    # Pass 2: 檔案層級（泛用處理；語言專屬在 definition_processor）
    # ─────────────────────────────────────────────────────────
    def process_generic_file(self, file_path: Path, file_name: str) -> None:
        """
        Process a generic (non-parseable) file and create appropriate nodes/relationships.

        設計重點：
        - File.path 一律用「絕對路徑」（與 definition_processor.py 的 DEFINED_IN 完全一致）
        - Folder/Package 的唯一鍵仍使用「相對路徑/qualified_name」
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        absolute_filepath = str(file_path)  # 絕對路徑
        relative_root = file_path.parent.relative_to(self.repo_path)  # 資料夾相對路徑

        # 取得「父容器」：Package（若 identify_structure 判為套件）或 Folder 或 Project
        parent_container_qn = self.structural_elements.get(relative_root)
        parent_label, parent_key, parent_val = (
            ("Package", "qualified_name", parent_container_qn)
            if parent_container_qn
            else (
                ("Folder", "path", str(relative_root))
                if relative_root != Path(".")
                else ("Project", "name", self.project_name)
            )
        )

        # 建立 File 節點（使用絕對路徑，與 DEFINED_IN 對齊）
        self.ingestor.ensure_node_batch(
            "File",
            {
                "path": absolute_filepath,
                "name": file_name,
                "extension": file_path.suffix,
            },
        )

        # 父容器 → File 關聯（Folder 用相對路徑、File 用絕對路徑）
        self.ingestor.ensure_relationship_batch(
            (parent_label, parent_key, parent_val),
            "CONTAINS_FILE",
            ("File", "path", absolute_filepath),
        )

        logger.debug(
            f"struct: {parent_label}({parent_val}) -[:CONTAINS_FILE]-> File({absolute_filepath})"
        )
