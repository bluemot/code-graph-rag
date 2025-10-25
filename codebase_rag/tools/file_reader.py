from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Tool


class FileReadResult(BaseModel):
    """Data model for file read results."""
    file_path: str
    content: str | None = None
    error_message: str | None = None


class FileReader:
    """Service to read file content from the filesystem.

    Additions:
      - read(file_path, start_line=None, end_line=None, allow_absolute_outside=True)
        Synchronous read with optional line slicing. Defaults to allowing absolute
        paths outside project_root (useful when graph stores absolute paths).
    """

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        # Define extensions that should be treated as binary and not read by this tool
        self.binary_extensions = {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".ico",
            ".tiff",
            ".webp",
        }
        logger.info(f"FileReader initialized with root: {self.project_root}")

    # -------------------------
    # Internal helper functions
    # -------------------------
    def _is_binary_ext(self, p: Path) -> bool:
        return p.suffix.lower() in self.binary_extensions

    def _resolve_path(
        self,
        file_path: str,
        allow_absolute_outside: bool = True,
    ) -> tuple[Optional[Path], Optional[str]]:
        """
        Resolve file path with security checks.

        If file_path is absolute and allow_absolute_outside is True, we allow it
        (with a warning) even if it's outside project_root. This is useful when
        the graph stores absolute file paths.

        Returns:
          (resolved_path, error_message)  -- only one of them will be non-None.
        """
        try:
            p = Path(file_path)
            if p.is_absolute():
                if allow_absolute_outside:
                    if not str(p).startswith(str(self.project_root)):
                        logger.warning(
                            f"[FileReader] Reading absolute path outside project root: {p}"
                        )
                    return p, None
                # Not allowed outside: enforce under project_root
                try:
                    p.relative_to(self.project_root)
                    return p, None
                except ValueError:
                    return None, "Security risk: Attempted to read file outside of project root."
            # relative path: anchor to project_root
            full = (self.project_root / p).resolve()
            try:
                full.relative_to(self.project_root)
            except ValueError:
                return None, "Security risk: Attempted to read file outside of project root."
            return full, None
        except Exception as e:
            logger.error(f"[FileReader] Failed to resolve path {file_path}: {e}")
            return None, f"An unexpected error occurred while resolving path: {e}"

    def _read_text(self, path: Path) -> tuple[Optional[str], Optional[str]]:
        """Read a text file with UTF-8, ignoring decode errors if needed."""
        if not path.is_file():
            return None, "File not found."

        if self._is_binary_ext(path):
            return (
                None,
                f"File '{path}' is a binary file. Use the 'analyze_document' tool for this file type.",
            )

        # Proceed with reading as a text file
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            return content, None
        except Exception as e:
            logger.error(f"[FileReader] Error reading file {path}: {e}")
            return None, f"An unexpected error occurred while reading file: {e}"

    # -------------------------
    # Public APIs
    # -------------------------
    def read(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        allow_absolute_outside: bool = True,
    ) -> str:
        """
        Synchronous file read with optional 1-based line slicing.

        Args:
          file_path: absolute or relative path. If absolute and outside project_root,
                     it's allowed by default (set allow_absolute_outside=False to forbid).
          start_line/end_line: 1-based inclusive line range; if omitted, returns whole file.
          allow_absolute_outside: allow reading absolute paths outside project_root.

        Returns:
          file content (or a slice). Empty string on error or if range invalid.
        """
        logger.info(f"[FileReader] read(path={file_path}, start={start_line}, end={end_line})")
        full_path, err = self._resolve_path(file_path, allow_absolute_outside=allow_absolute_outside)
        if err:
            logger.warning(f"[FileReader] {err} (path={file_path})")
            return ""

        assert full_path is not None
        content, err2 = self._read_text(full_path)
        if err2 or content is None:
            logger.warning(f"[FileReader] {err2} (path={file_path})")
            return ""

        if start_line is None and end_line is None:
            return content

        # Slice by lines (1-based)
        lines = content.splitlines()
        n = len(lines)
        s = 1 if start_line is None else max(1, int(start_line))
        e = n if end_line is None else max(1, int(end_line))
        if n == 0 or e < s:
            return ""
        s = min(s, n)
        e = min(e, n)
        return "\n".join(lines[s - 1 : e])

    async def read_file(self, file_path: str) -> FileReadResult:
        """Reads and returns the content of a text-based file (whole file)."""
        logger.info(f"[FileReader] Attempting to read file: {file_path}")

        full_path, err = self._resolve_path(
            file_path, allow_absolute_outside=False  # tool API keeps strict default
        )
        if err:
            return FileReadResult(file_path=file_path, error_message=err)

        assert full_path is not None
        content, err2 = self._read_text(full_path)
        if err2:
            logger.warning(f"[FileReader] {err2}")
            return FileReadResult(file_path=file_path, error_message=err2)

        logger.info(f"[FileReader] Successfully read text from {file_path}")
        return FileReadResult(file_path=file_path, content=content)


def create_file_reader_tool(file_reader: FileReader) -> Tool:
    """Factory function to create the file reader tool."""

    async def read_file_content(file_path: str) -> str:
        """
        Reads the content of a specified text-based file (e.g., source code, README.md, config files).
        This tool should NOT be used for binary files like PDFs or images. For those, use the 'analyze_document' tool.
        """
        result = await file_reader.read_file(file_path)
        if result.error_message:
            return f"Error: {result.error_message}"
        return result.content or ""

    return Tool(
        function=read_file_content,
        description="Reads the content of text-based files. For documents like PDFs or images, use the 'analyze_document' tool instead.",
    )
