# --- main.py (merged with resumable Qdrant upsert, wait=False, retries) ---
# [全部原有 import 保留；僅確保有 time/hashlib/json/os 等]
import asyncio
import json
import re
import shlex
import shutil
import sys
import uuid
import os
import math
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional

import typer
from loguru import logger
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from .config import (
    EDIT_INDICATORS,
    EDIT_REQUEST_KEYWORDS,
    EDIT_TOOLS,
    ORANGE_STYLE,
    settings,
)
from .graph_updater import GraphUpdater, MemgraphIngestor
from .parser_loader import load_parsers
from .services.llm import CypherGenerator, create_rag_orchestrator
from .tools.code_retrieval import CodeRetriever, create_code_retrieval_tool
from .tools.codebase_query import create_query_tool
from .tools.directory_lister import DirectoryLister, create_directory_lister_tool
from .tools.document_analyzer import DocumentAnalyzer, create_document_analyzer_tool
from .tools.file_editor import FileEditor, create_file_editor_tool
from .tools.file_reader import FileReader, create_file_reader_tool
from .tools.file_writer import FileWriter, create_file_writer_tool
from .tools.shell_command import ShellCommander, create_shell_command_tool

# Optional (lazy) deps for Qdrant + embeddings
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except Exception:
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore
    PointStruct = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

try:
    import requests
except Exception:
    requests = None  # type: ignore

try:
    import openai
except Exception:
    openai = None  # type: ignore

# Style constants
confirm_edits_globally = True

# Pre-compile regex patterns
_FILE_MODIFICATION_PATTERNS = [
    re.compile(
        r"(modified|updated|created|edited):\s*[\w/\\.-]+\.(py|js|ts|java|cpp|c|h|go|rs)"
    ),
    re.compile(
        r"file\s+[\w/\\.-]+\.(py|js|ts|java|cpp|c|h|go|rs)\s+(modified|updated|created|edited)"
    ),
    re.compile(r"writing\s+to\s+[\w/\\.-]+\.(py|js|ts|java|cpp|c|h|go|rs)"),
]


app = typer.Typer(
    name="graph-code",
    help="An accurate Retrieval-Augmented Generation (RAG) system that analyzes "
    "multi-language codebases using Tree-sitter, builds comprehensive knowledge "
    "graphs, and enables natural language querying of codebase structure and "
    "relationships.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(width=None, force_terminal=True)

# Session logging
session_log_file = None
session_cancelled = False


def init_session_log(project_root: Path) -> Path:
    global session_log_file
    log_dir = project_root / ".tmp"
    log_dir.mkdir(exist_ok=True)
    session_log_file = log_dir / f"session_{uuid.uuid4().hex[:8]}.log"
    with open(session_log_file, "w") as f:
        f.write("=== CODE-GRAPH RAG SESSION LOG ===\n\n")
    return session_log_file


def log_session_event(event: str) -> None:
    global session_log_file
    if session_log_file:
        with open(session_log_file, "a") as f:
            f.write(f"{event}\n")


def get_session_context() -> str:
    global session_log_file
    if session_log_file and session_log_file.exists():
        content = Path(session_log_file).read_text()
        return f"\n\n[SESSION CONTEXT - Previous conversation in this session]:\n{content}\n[END SESSION CONTEXT]\n\n"
    return ""


def is_edit_operation_request(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in EDIT_REQUEST_KEYWORDS)


async def _handle_rejection(
    rag_agent: Any, message_history: list[Any], console: Console
) -> Any:
    rejection_message = "The user has rejected the changes that were made. Please acknowledge this and consider if any changes need to be reverted."

    with console.status("[bold yellow]Processing rejection...[/bold yellow]"):
        rejection_response = await run_with_cancellation(
            console,
            rag_agent.run(rejection_message, message_history=message_history),
        )

    if not (
        isinstance(rejection_response, dict) and rejection_response.get("cancelled")
    ):
        rejection_markdown = Markdown(rejection_response.output)
        console.print(
            Panel(
                rejection_markdown,
                title="[bold yellow]Response to Rejection[/bold yellow]",
                border_style="yellow",
            )
        )
        message_history.extend(rejection_response.new_messages())

    return rejection_response


def is_edit_operation_response(response_text: str) -> bool:
    response_lower = response_text.lower()
    tool_usage = any(tool in response_lower for tool in EDIT_TOOLS)
    content_indicators = any(
        indicator in response_lower for indicator in EDIT_INDICATORS
    )
    pattern_match = any(
        pattern.search(response_lower) for pattern in _FILE_MODIFICATION_PATTERNS
    )
    return tool_usage or content_indicators or pattern_match


def _setup_common_initialization(repo_path: str) -> Path:
    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}")
    project_root = Path(repo_path).resolve()
    tmp_dir = project_root / ".tmp"
    if tmp_dir.exists():
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)
        else:
            tmp_dir.unlink()
    tmp_dir.mkdir()
    return project_root


def _create_configuration_table(
    repo_path: str,
    title: str = "Graph-Code Initializing...",
    language: str | None = None,
) -> Table:
    table = Table(title=f"[bold green]{title}[/bold green]")
    table.add_column("Configuration", style="cyan")
    table.add_column("Value", style="magenta")

    orchestrator_config = settings.active_orchestrator_config
    table.add_row(
        "Orchestrator Model",
        f"{orchestrator_config.model_id} ({orchestrator_config.provider})",
    )

    cypher_config = settings.active_cypher_config
    table.add_row(
        "Cypher Model", f"{cypher_config.model_id} ({cypher_config.provider})"
    )

    orch_endpoint = (
        orchestrator_config.endpoint
        if orchestrator_config.provider == "ollama"
        else None
    )
    cypher_endpoint = (
        cypher_config.endpoint if cypher_config.provider == "ollama" else None
    )

    if orch_endpoint and cypher_endpoint and orch_endpoint == cypher_endpoint:
        table.add_row("Ollama Endpoint", orch_endpoint)
    else:
        if orch_endpoint:
            table.add_row("Ollama Endpoint (Orchestrator)", orch_endpoint)
        if cypher_endpoint:
            table.add_row("Ollama Endpoint (Cypher)", cypher_endpoint)

    confirmation_status = (
        "Enabled" if confirm_edits_globally else "Disabled (YOLO Mode)"
    )
    table.add_row("Edit Confirmation", confirmation_status)
    table.add_row("Target Repository", repo_path)
    if language:
        table.add_row("Target Language", language)

    return table


async def run_optimization_loop(
    rag_agent: Any,
    message_history: list[Any],
    project_root: Path,
    language: str,
    reference_document: str | None = None,
) -> None:
    global session_cancelled

    init_session_log(project_root)
    console.print(
        f"[bold green]Starting {language} optimization session...[/bold green]"
    )
    document_info = (
        f" using the reference document: {reference_document}"
        if reference_document
        else ""
    )
    console.print(
        Panel(
            f"[bold yellow]The agent will analyze your codebase{document_info} and propose specific optimizations."
            f" You'll be asked to approve each suggestion before implementation."
            f" Type 'exit' or 'quit' to end the session.[/bold yellow]",
            border_style="yellow",
        )
    )

    instructions = [
        "Use your code retrieval and graph querying tools to understand the codebase structure",
        "Read relevant source files to identify optimization opportunities",
    ]
    if reference_document:
        instructions.append(
            f"Use the analyze_document tool to reference best practices from {reference_document}"
        )
    instructions.extend(
        [
            f"Reference established patterns and best practices for {language}",
            "Propose specific, actionable optimizations with file references",
            "IMPORTANT: Do not make any changes yet - just propose them and wait for approval",
            "After approval, use your file editing tools to implement the changes",
        ]
    )

    numbered_instructions = "\n".join(
        f"{i + 1}. {inst}" for i, inst in enumerate(instructions)
    )

    initial_question = f"""
I want you to analyze my {language} codebase and propose specific optimizations based on best practices.

Please:
{numbered_instructions}

Start by analyzing the codebase structure and identifying the main areas that could benefit from optimization.
Remember: Propose changes first, wait for my approval, then implement.
"""

    first_run = True
    question = initial_question

    while True:
        try:
            if not first_run:
                question = await asyncio.to_thread(
                    get_multiline_input, "[bold cyan]Your response[/bold cyan]"
                )

            if question.lower() in ["exit", "quit"]:
                break
            if not question.strip():
                continue

            log_session_event(f"USER: {question}")

            if session_cancelled:
                question_with_context = question + get_session_context()
                session_cancelled = False
            else:
                question_with_context = question

            question_with_context = _handle_chat_images(
                question_with_context, project_root
            )

            with console.status(
                "[bold green]Agent is analyzing codebase... (Press Ctrl+C to cancel)[/bold green]"
            ):
                response = await run_with_cancellation(
                    console,
                    rag_agent.run(
                        question_with_context, message_history=message_history
                    ),
                )

                if isinstance(response, dict) and response.get("cancelled"):
                    log_session_event("ASSISTANT: [Analysis was cancelled]")
                    session_cancelled = True
                    continue

            markdown_response = Markdown(response.output)
            console.print(
                Panel(
                    markdown_response,
                    title="[bold green]Optimization Agent[/bold green]",
                    border_style="green",
                )
            )

            if confirm_edits_globally and is_edit_operation_response(response.output):
                console.print(
                    "\n[bold yellow]⚠️  This optimization has performed file modifications.[/bold yellow]"
                )
                if not Confirm.ask(
                    "[bold cyan]Do you want to keep these optimizations?[/bold cyan]"
                ):
                    console.print(
                        "[bold red]❌ Optimizations rejected by user.[/bold red]"
                    )
                    await _handle_rejection(rag_agent, message_history, console)
                    first_run = False
                    continue
                else:
                    console.print(
                        "[bold green]✅ Optimizations approved by user.[/bold green]"
                    )

            log_session_event(f"ASSISTANT: {response.output}")
            message_history.extend(response.new_messages())
            first_run = False

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("1.An unexpected error occurred: {}", e, exc_info=True)
            console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")


async def run_with_cancellation(
    console: Console, coro: Any, timeout: float | None = None
) -> Any:
    task = asyncio.create_task(coro)
    try:
        return await asyncio.wait_for(task, timeout=timeout) if timeout else await task
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        console.print(
            f"\n[bold yellow]Operation timed out after {timeout} seconds.[/bold yellow]"
        )
        return {"cancelled": True, "timeout": True}
    except (asyncio.CancelledError, KeyboardInterrupt):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        console.print("\n[bold yellow]Thinking cancelled.[/bold yellow]")
        return {"cancelled": True}


def _handle_chat_images(question: str, project_root: Path) -> str:
    try:
        tokens = shlex.split(question)
    except ValueError:
        tokens = question.split()

    image_extensions = (".png", ".jpg", ".jpeg", ".gif")
    image_files = [
        token
        for token in tokens
        if token.startswith("/") and token.lower().endswith(image_extensions)
    ]

    if not image_files:
        return question

    updated_question = question
    tmp_dir = project_root / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    for original_path_str in image_files:
        original_path = Path(original_path_str)

        if not original_path.exists() or not original_path.is_file():
            logger.warning(f"Image path found, but does not exist: {original_path_str}")
            continue

        try:
            new_path = tmp_dir / f"{uuid.uuid4()}-{original_path.name}"
            shutil.copy(original_path, new_path)
            new_relative_path = new_path.relative_to(project_root)

            path_variants = [
                original_path_str.replace(" ", r"\ "),
                f"'{original_path_str}'",
                f'"{original_path_str}"',
                original_path_str,
            ]

            replaced = False
            for variant in path_variants:
                if variant in updated_question:
                    updated_question = updated_question.replace(
                        variant, str(new_relative_path)
                    )
                    replaced = True
                    break

            if not replaced:
                logger.warning(
                    f"Could not find original path in question for replacement: {original_path_str}"
                )

            logger.info(f"Copied image to temporary path: {new_relative_path}")
        except Exception as e:
            logger.error(f"Failed to copy image to temporary directory: {e}")

    return updated_question


def get_multiline_input(prompt_text: str = "Ask a question") -> str:
    bindings = KeyBindings()

    @bindings.add("c-j")
    def submit(event: Any) -> None:
        event.app.exit(result=event.app.current_buffer.text)

    @bindings.add("enter")
    def new_line(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def keyboard_interrupt(event: Any) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    clean_prompt = Text.from_markup(prompt_text).plain

    print_formatted_text(
        HTML(
            f"<ansigreen><b>{clean_prompt}</b></ansigreen> <ansiyellow>(Press Ctrl+J to submit, Enter for new line)</ansiyellow>: "
        )
    )

    result = prompt(
        "",
        multiline=True,
        key_bindings=bindings,
        wrap_lines=True,
        style=ORANGE_STYLE,
    )
    if result is None:
        raise EOFError
    return result.strip()  # type: ignore[no-any-return]


async def run_chat_loop(
    rag_agent: Any, message_history: list[Any], project_root: Path
) -> None:
    global session_cancelled
    init_session_log(project_root)

    while True:
        try:
            question = await asyncio.to_thread(
                get_multiline_input, "[bold cyan]Ask a question[/bold cyan]"
            )

            if question.lower() in ["exit", "quit"]:
                break
            if not question.strip():
                continue

            log_session_event(f"USER: {question}")

            if session_cancelled:
                question_with_context = question + get_session_context()
                session_cancelled = False
            else:
                question_with_context = question

            question_with_context = _handle_chat_images(
                question_with_context, project_root
            )

            might_edit = is_edit_operation_request(question)
            if confirm_edits_globally and might_edit:
                console.print(
                    "\n[bold yellow]⚠️  This request might result in file modifications.[/bold yellow]"
                )
                if not Confirm.ask(
                    "[bold cyan]Do you want to proceed with this request?[/bold cyan]"
                ):
                    console.print("[bold red]❌ Request cancelled by user.[/bold red]")
                    continue

            with console.status(
                "[bold green]Thinking... (Press Ctrl+C to cancel)[/bold green]"
            ):
                response = await run_with_cancellation(
                    console,
                    rag_agent.run(
                        question_with_context, message_history=message_history
                    ),
                )

                if isinstance(response, dict) and response.get("cancelled"):
                    log_session_event("ASSISTANT: [Thinking was cancelled]")
                    session_cancelled = True
                    continue

            markdown_response = Markdown(response.output)
            console.print(
                Panel(
                    markdown_response,
                    title="[bold green]Assistant[/bold green]",
                    border_style="green",
                )
            )

            if confirm_edits_globally and is_edit_operation_response(response.output):
                console.print(
                    "\n[bold yellow]⚠️  The assistant has performed file modifications.[/bold yellow]"
                )

                if not Confirm.ask(
                    "[bold cyan]Do you want to keep these changes?[/bold cyan]"
                ):
                    console.print("[bold red]❌ User rejected the changes.[/bold red]")
                    await _handle_rejection(rag_agent, message_history, console)
                    continue
                else:
                    console.print(
                        "[bold green]✅ Changes accepted by user.[/bold green]"
                    )

            log_session_event(f"ASSISTANT: {response.output}")
            message_history.extend(response.new_messages())

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("2.An unexpected error occurred: {}", e, exc_info=True)
            console.print(f"[bold red]An unexpected error occurred: {e}[/bold red]")


def _update_single_model_setting(role: str, model_string: str) -> None:
    provider, model = settings.parse_model_string(model_string)
    if role == "orchestrator":
        current_config = settings.active_orchestrator_config
        set_method = settings.set_orchestrator
    else:
        current_config = settings.active_cypher_config
        set_method = settings.set_cypher

    kwargs = {
        "api_key": current_config.api_key,
        "endpoint": current_config.endpoint,
        "project_id": current_config.project_id,
        "region": current_config.region,
        "provider_type": current_config.provider_type,
        "thinking_budget": current_config.thinking_budget,
        "service_account_file": current_config.service_account_file,
    }

    if provider == "ollama" and not kwargs["endpoint"]:
        kwargs["endpoint"] = str(settings.LOCAL_MODEL_ENDPOINT)
        kwargs["api_key"] = "ollama"

    set_method(provider, model, **kwargs)


def _update_model_settings(
    orchestrator: str | None,
    cypher: str | None,
) -> None:
    if orchestrator:
        _update_single_model_setting("orchestrator", orchestrator)
    if cypher:
        _update_single_model_setting("cypher", cypher)


def _export_graph_to_file(ingestor: MemgraphIngestor, output: str) -> bool:
    try:
        graph_data = ingestor.export_graph_to_dict()
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        console.print(
            f"[bold green]Graph exported successfully to: {output_path.absolute()}[/bold green]"
        )
        console.print(
            f"[bold cyan]Export contains {graph_data['metadata']['total_nodes']} nodes and {graph_data['metadata']['total_relationships']} relationships[/bold cyan]"
        )
        return True

    except Exception as e:
        console.print(f"[bold red]Failed to export graph: {e}[/bold red]")
        logger.error(f"Export error: {e}", exc_info=True)
        return False


def _initialize_services_and_agent(repo_path: str, ingestor: MemgraphIngestor) -> Any:
    from .providers.base import get_provider

    def _validate_provider_config(role: str, config: Any) -> None:
        try:
            provider = get_provider(
                config.provider,
                api_key=config.api_key,
                endpoint=config.endpoint,
                project_id=config.project_id,
                region=config.region,
                provider_type=config.provider_type,
                thinking_budget=config.thinking_budget,
                service_account_file=config.service_account_file,
            )
            provider.validate_config()
        except Exception as e:
            raise ValueError(f"{role.title()} configuration error: {e}") from e

    _validate_provider_config("orchestrator", settings.active_orchestrator_config)
    _validate_provider_config("cypher", settings.active_cypher_config)

    cypher_generator = CypherGenerator()
    code_retriever = CodeRetriever(project_root=repo_path, ingestor=ingestor)
    file_reader = FileReader(project_root=repo_path)
    file_writer = FileWriter(project_root=repo_path)
    file_editor = FileEditor(project_root=repo_path)
    shell_commander = ShellCommander(
        project_root=repo_path, timeout=settings.SHELL_COMMAND_TIMEOUT
    )
    directory_lister = DirectoryLister(project_root=repo_path)
    document_analyzer = DocumentAnalyzer(project_root=repo_path)

    query_tool = create_query_tool(ingestor, cypher_generator, console)
    code_tool = create_code_retrieval_tool(code_retriever)
    file_reader_tool = create_file_reader_tool(file_reader)
    file_writer_tool = create_file_writer_tool(file_writer)
    file_editor_tool = create_file_editor_tool(file_editor)
    shell_command_tool = create_shell_command_tool(shell_commander)
    directory_lister_tool = create_directory_lister_tool(directory_lister)
    document_analyzer_tool = create_document_analyzer_tool(document_analyzer)

    rag_agent = create_rag_orchestrator(
        tools=[
            query_tool,
            code_tool,
            file_reader_tool,
            file_writer_tool,
            file_editor_tool,
            shell_command_tool,
            directory_lister_tool,
            document_analyzer_tool,
        ]
    )
    return rag_agent


async def main_async(repo_path: str, batch_size: int) -> None:
    project_root = _setup_common_initialization(repo_path)
    table = _create_configuration_table(repo_path)
    console.print(table)

    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=batch_size,
    ) as ingestor:
        console.print("[bold green]Successfully connected to Memgraph.[/bold green]")
        console.print(
            Panel(
                "[bold yellow]Ask questions about your codebase graph. Type 'exit' or 'quit' to end.[/bold yellow]",
                border_style="yellow",
            )
        )

        rag_agent = _initialize_services_and_agent(repo_path, ingestor)
        await run_chat_loop(rag_agent, [], project_root)


@app.command()
def start(
    repo_path: str | None = typer.Option(
        None, "--repo-path", help="Path to the target repository for code retrieval"
    ),
    update_graph: bool = typer.Option(
        False,
        "--update-graph",
        help="Update the knowledge graph by parsing the repository",
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help="Clean the database before updating (use when adding first repo)",
    ),
    output: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help="Export graph to JSON file after updating (requires --update-graph)",
    ),
    orchestrator: str | None = typer.Option(
        None,
        "--orchestrator",
        help="Specify orchestrator as provider:model (e.g., ollama:llama3.2, openai:gpt-4, google:gemini-2.5-pro)",
    ),
    cypher: str | None = typer.Option(
        None,
        "--cypher",
        help="Specify cypher model as provider:model (e.g., ollama:codellama, google:gemini-2.5-flash)",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Disable confirmation prompts for edit operations (YOLO mode)",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Number of buffered nodes/relationships before flushing to Memgraph",
    ),
) -> None:
    global confirm_edits_globally
    confirm_edits_globally = not no_confirm

    target_repo_path = repo_path or settings.TARGET_REPO_PATH

    if output and not update_graph:
        console.print(
            "[bold red]Error: --output/-o option requires --update-graph to be specified.[/bold red]"
        )
        raise typer.Exit(1)

    _update_model_settings(orchestrator, cypher)
    effective_batch_size = settings.resolve_batch_size(batch_size)

    if update_graph:
        repo_to_update = Path(target_repo_path)
        console.print(
            f"[bold green]Updating knowledge graph for: {repo_to_update}[/bold green]"
        )

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=effective_batch_size,
        ) as ingestor:
            if clean:
                console.print("[bold yellow]Cleaning database...[/bold yellow]")
                ingestor.clean_database()
            ingestor.ensure_constraints()
            parsers, queries = load_parsers()
            updater = GraphUpdater(ingestor, repo_to_update, parsers, queries)
            updater.run()

            if output:
                console.print(f"[bold cyan]Exporting graph to: {output}[/bold cyan]")
                if not _export_graph_to_file(ingestor, output):
                    raise typer.Exit(1)

        console.print("[bold green]Graph update completed![/bold green]")
        return

    try:
        asyncio.run(main_async(target_repo_path, effective_batch_size))
    except KeyboardInterrupt:
        console.print("\n[bold red]Application terminated by user.[/bold red]")
    except ValueError as e:
        console.print(f"[bold red]Startup Error: {e}[/bold red]")


@app.command()
def export(
    output: str = typer.Option(
        ..., "-o", "--output", help="Output file path for the exported graph"
    ),
    format_json: bool = typer.Option(
        True, "--json/--no-json", help="Export in JSON format"
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Number of buffered nodes/relationships before flushing to Memgraph",
    ),
) -> None:
    if not format_json:
        console.print(
            "[bold red]Error: Currently only JSON format is supported.[/bold red]"
        )
        raise typer.Exit(1)

    console.print("[bold cyan]Connecting to Memgraph to export graph...[/bold cyan]")
    effective_batch_size = settings.resolve_batch_size(batch_size)

    try:
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=effective_batch_size,
        ) as ingestor:
            console.print("[bold cyan]Exporting graph data...[/bold cyan]")
            if not _export_graph_to_file(ingestor, output):
                raise typer.Exit(1)

    except Exception as e:
        console.print(f"[bold red]Failed to export graph: {e}[/bold red]")
        logger.error(f"Export error: {e}", exc_info=True)
        raise typer.Exit(1) from e


async def main_optimize_async(
    language: str,
    target_repo_path: str,
    reference_document: str | None = None,
    orchestrator: str | None = None,
    cypher: str | None = None,
    batch_size: int | None = None,
) -> None:
    project_root = _setup_common_initialization(target_repo_path)
    _update_model_settings(orchestrator, cypher)

    console.print(
        f"[bold cyan]Initializing optimization session for {language} codebase: {project_root}[/bold cyan]"
    )
    table = _create_configuration_table(
        str(project_root), "Optimization Session Configuration", language
    )
    console.print(table)

    effective_batch_size = settings.resolve_batch_size(batch_size)

    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=effective_batch_size,
    ) as ingestor:
        console.print("[bold green]Successfully connected to Memgraph.[/bold green]")
        rag_agent = _initialize_services_and_agent(target_repo_path, ingestor)
        await run_optimization_loop(
            rag_agent, [], project_root, language, reference_document
        )


@app.command()
def optimize(
    language: str = typer.Argument(
        ...,
        help="Programming language to optimize for (e.g., python, java, javascript, cpp)",
    ),
    repo_path: str | None = typer.Option(
        None, "--repo-path", help="Path to the repository to optimize"
    ),
    reference_document: str | None = typer.Option(
        None,
        "--reference-document",
        help="Path to reference document/book for optimization guidance",
    ),
    orchestrator: str | None = typer.Option(
        None,
        "--orchestrator",
        help="Specify orchestrator as provider:model (e.g., ollama:llama3.2, openai:gpt-4, google:gemini-2.5-pro)",
    ),
    cypher: str | None = typer.Option(
        None,
        "--cypher",
        help="Specify cypher model as provider:model (e.g., ollama:codellama, google:gemini-2.5-flash)",
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help="Disable confirmation prompts for edit operations (YOLO mode)",
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help="Number of buffered nodes/relationships before flushing to Memgraph",
    ),
) -> None:
    global confirm_edits_globally
    confirm_edits_globally = not no_confirm
    target_repo_path = repo_path or settings.TARGET_REPO_PATH

    try:
        asyncio.run(
            main_optimize_async(
                language,
                target_repo_path,
                reference_document,
                orchestrator,
                cypher,
                batch_size,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[bold red]Optimization session terminated by user.[/bold red]")
    except ValueError as e:
        console.print(f"[bold red]Startup Error: {e}[/bold red]")


# =============================
# push_code_to_qdrant_env 命令
# =============================

class _Embedder:
    def __init__(self, backend: str, model: str):
        self.backend = (backend or "SENTENCE_TRANSFORMERS").upper()
        self.model = model or "all-MiniLM-L6-v2"
        if self.backend == "SENTENCE_TRANSFORMERS":
            if SentenceTransformer is None:
                raise RuntimeError("sentence-transformers is not installed")
            self._st = SentenceTransformer(self.model, device=os.environ.get("EMBEDDING_DEVICE", None))
        elif self.backend == "OLLAMA":
            if requests is None:
                raise RuntimeError("requests is required for OLLAMA backend")
            self._ollama = os.getenv("OLLAMA_ENDPOINT", str(settings.EMBEDDING_ENDPOINT))
        elif self.backend == "OPENAI":
            if openai is None:
                raise RuntimeError("openai package is not installed")
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY not set")
            openai.api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise RuntimeError(f"Unknown EMBEDDING_BACKEND: {backend}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.backend == "SENTENCE_TRANSFORMERS":
            vecs = self._st.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vecs]
        if self.backend == "OLLAMA":
            out: List[List[float]] = []
            for t in texts:
                r = requests.post(
                    f"{self._ollama}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                    timeout=120,
                )
                r.raise_for_status()
                out.append(r.json()["embedding"])  # type: ignore
            return out
        if self.backend == "OPENAI":
            resp = openai.embeddings.create(model=self.model, input=texts)  # type: ignore
            return [d["embedding"] if isinstance(d, dict) else d.embedding for d in resp.data]
        raise AssertionError("unreachable")


def _qdrant_ensure_collection(client: QdrantClient, name: str, dim: int, distance: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        return
    dist = Distance.COSINE if (distance or "cosine").lower() == "cosine" else Distance.DOT
    client.recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=dist),
    )


def _guess_lang_by_ext(path: str) -> str:
    p = path.lower()
    if p.endswith((".c", ".h")): return "c"
    if p.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx")): return "cpp"
    if p.endswith((".py",)): return "python"
    if p.endswith((".js",)): return "javascript"
    if p.endswith((".ts",)): return "typescript"
    if p.endswith((".java",)): return "java"
    if p.endswith((".go",)): return "go"
    if p.endswith((".rs",)): return "rust"
    if p.endswith((".lua",)): return "lua"
    return "unknown"


def _mg_count_functions(ing: MemgraphIngestor) -> int:
    rows = ing.fetch_all(
        """
        MATCH (f:Function)-[:DEFINED_IN]->(file:File)
        WHERE f.start_line IS NOT NULL AND f.end_line IS NOT NULL AND file.path IS NOT NULL
        RETURN count(f) AS n
        """
    )
    if not rows:
        return 0
    r0 = rows[0]
    if isinstance(r0, dict) and "n" in r0:
        return int(r0["n"])
    if isinstance(r0, (list, tuple)) and r0:
        return int(r0[0])
    try:
        return int(list(r0.values())[0])  # type: ignore
    except Exception:
        return 0


def _mg_iter_pages(ing: MemgraphIngestor, limit: int) -> Iterable[List[Dict[str, Any]]]:
    total = _mg_count_functions(ing)
    if total == 0:
        yield []
        return
    pages = math.ceil(total / limit)
    for i in range(pages):
        skip = i * limit
        rows = ing.fetch_all(
            """
            MATCH (f:Function)-[:DEFINED_IN]->(file:File)
            WHERE f.start_line IS NOT NULL AND f.end_line IS NOT NULL AND file.path IS NOT NULL
            RETURN f.qualified_name AS qn, f.name AS name, file.path AS path, f.start_line AS start, f.end_line AS end
            ORDER BY qn
            SKIP $skip LIMIT $limit
            """,
            {"skip": skip, "limit": limit},
        )
        norm: List[Dict[str, Any]] = []
        for r in rows or []:
            if isinstance(r, dict):
                norm.append({
                    "qn": r.get("qn"),
                    "name": r.get("name"),
                    "path": r.get("path"),
                    "start": r.get("start"),
                    "end": r.get("end"),
                })
            else:
                try:
                    qn, name, path, start, end = r
                    norm.append({"qn": qn, "name": name, "path": path, "start": start, "end": end})
                except Exception:
                    continue
        yield norm


@app.command("push-code-to-qdrant")
def push_code_to_qdrant_env(
    repo_path: str | None = typer.Option(
        None, "--repo-path", help="Path to the repository to optimize"
    ),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of functions to push (debug)")
) -> None:
    """將 (Function -> File path/start/end) 的程式碼切片，嵌入並寫入 Qdrant。設定全部來自 .env/settings。"""
    if QdrantClient is None:
        console.print("[bold red]qdrant-client is not installed.[/bold red]")
        raise typer.Exit(1)

    # ---- Env/Settings
    repo_path = repo_path or settings.TARGET_REPO_PATH
    repo_root = Path(repo_path).resolve()
    mg_host = settings.MEMGRAPH_HOST
    mg_port = settings.MEMGRAPH_PORT

    q_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    q_api = os.getenv("QDRANT_API_KEY", "")
    q_collection = os.getenv("QDRANT_COLLECTION", "code_chunks")
    q_distance = os.getenv("QDRANT_DISTANCE", "Cosine")
    q_batch = int(os.getenv("QDRANT_BATCH", "64"))
    max_lines = int(os.getenv("QDRANT_MAX_LINES_PER_CHUNK", "400"))
    q_timeout = float(os.getenv("QDRANT_TIMEOUT", "60"))
    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    resume_path = os.getenv("QDRANT_RESUME_STATE", ".tmp/qdrant_resume.json")

    emb_backend = os.getenv("EMBEDDING_BACKEND", "SENTENCE_TRANSFORMERS")
    emb_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    file_reader = FileReader(project_root=str(repo_root))

    # ---- Init embedder and qdrant
    embedder = _Embedder(emb_backend, emb_model)
    dim = len(embedder.embed(["dim-probe"])[0])

    qdr = QdrantClient(
        url=q_url,
        api_key=q_api or None,
        timeout=q_timeout,
        prefer_grpc=prefer_grpc,
    )
    _qdrant_ensure_collection(qdr, q_collection, dim, q_distance)

    # ---- Resume helpers
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)

    def load_resume():
        try:
            with open(resume_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"page": 0, "offset_in_page": 0}

    def save_resume(state):
        tmp = resume_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, resume_path)

    state = load_resume()
    start_page = int(state.get("page", 0))
    start_offset = int(state.get("offset_in_page", 0))

    # ---- upsert with retry (wait=False)
    def upsert_with_retry(points: List[PointStruct], max_retries=3):
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                qdr.upsert(collection_name=q_collection, points=points, wait=False)
                return True
            except Exception as e:
                if attempt == max_retries:
                    console.print(f"[red]Upsert failed after retries: {e}[/red]")
                    return False
                time.sleep(backoff)
                backoff *= 2

    # ---- Iterate graph (paged)
    sent = 0
    processed = 0
    t0 = time.time()

    def _chunk_fn(text: str, start: int, end: int, win: int, step: int):
        lines = text.splitlines()
        n = len(lines)
        if n <= win:
            yield (start, end, text)
            return
        cur = 0
        while cur < n:
            seg = lines[cur:cur+win]
            seg_start = start + cur
            seg_end = min(start + cur + len(seg) - 1, end)
            yield (seg_start, seg_end, "\n".join(seg))
            if cur + win >= n:
                break
            cur += step

    with MemgraphIngestor(host=mg_host, port=mg_port, batch_size=1000) as ing:
        total = _mg_count_functions(ing)
        if limit and limit > 0:
            total = min(total, limit)
        console.print(f"[bold cyan]Preparing to push ~{total} functions to Qdrant...[/bold cyan]")

        for page_index, page in enumerate(_mg_iter_pages(ing, limit=1000)):
            # 跳過已完成的頁
            if page_index < start_page:
                continue
            if not page:
                # 沒資料也視為完成一頁
                save_resume({"page": page_index + 1, "offset_in_page": 0})
                continue
            if limit and processed >= limit:
                break

            texts: List[str] = []
            metas: List[Dict[str, Any]] = []

            inpage_start = start_offset if page_index == start_page else 0

            for idx_in_page, r in enumerate(page):
                if idx_in_page < inpage_start:
                    continue
                if limit and processed >= limit:
                    break

                qn = str(r["qn"]) if r.get("qn") is not None else ""
                path = str(r["path"]) if r.get("path") is not None else ""
                start = int(r["start"] or 0)
                end = int(r["end"] or 0)
                if not path or not start or not end or end < start:
                    save_resume({"page": page_index, "offset_in_page": idx_in_page + 1})
                    continue

                try:
                    text = file_reader.read(path, start_line=start, end_line=end)
                except Exception as e:
                    console.print(f"[yellow]Skip file (read failed): {path} ({e})[/yellow]")
                    save_resume({"page": page_index, "offset_in_page": idx_in_page + 1})
                    continue

                for seg_s, seg_e, seg_text in _chunk_fn(text, start, end, max_lines, max(max_lines//2, 50)):
                    if not seg_text.strip():
                        continue
                    payload = {
                        "repo": repo_root.name,
                        "language": _guess_lang_by_ext(path),
                        "path": path,
                        "start": seg_s,
                        "end": seg_e,
                        "qualified_name": qn,
                        "kind": "function",
                    }
                    texts.append(seg_text)
                    metas.append(payload)

                # 完成此筆（無論是否產生段落）→ 前進游標
                save_resume({"page": page_index, "offset_in_page": idx_in_page + 1})
                processed += 1

            if not texts:
                # 這一頁沒有任何文本 → 前進到下一頁，offset 歸零
                save_resume({"page": page_index + 1, "offset_in_page": 0})
                continue

            # 嵌入 + 小批次 upsert（非阻塞）
            vecs = embedder.embed(texts)
            sub = max(1, q_batch)
            sent_this_page = 0

            # 準備所有 points
            points: List[PointStruct] = []
            for payload, vec, text in zip(metas, vecs, texts):
                pid = int(hashlib.sha256(
                    f"{payload['path']}:{payload['start']}-{payload['end']}:{payload['qualified_name']}".encode()
                ).hexdigest()[:16], 16)
                points.append(PointStruct(id=pid, vector=vec, payload={**payload, "code": text}))

            # 送出（分段 + 重試）
            for i in range(0, len(points), sub):
                chunk = points[i:i+sub]
                ok = upsert_with_retry(chunk, max_retries=3)
                if ok:
                    sent += len(chunk)
                    sent_this_page += len(chunk)

            console.print(f"[green]Page {page_index}: upserted {sent_this_page} points (total {sent})[/green]")

            # 這一頁完成 → 前進到下一頁，offset 歸零
            save_resume({"page": page_index + 1, "offset_in_page": 0})

    t1 = time.time()
    console.print(f"[bold green]Done. Upserted {sent} points in {t1 - t0:.1f}s[/bold green]")

# =============================
# push_stmts_to_qdrant 指令（把 Stmt.src 丟向量庫）
# =============================

def _mg_count_stmts(ing: MemgraphIngestor) -> int:
    rows = ing.fetch_all(
        """
        MATCH (s:Stmt)<-[:CONTAINS_STMT]-(:BasicBlock)<-[:CONTAINS_BLOCK]-(:Function)
        RETURN count(s) AS n
        """
    )
    if not rows:
        return 0
    r0 = rows[0]
    if isinstance(r0, dict) and "n" in r0:
        return int(r0["n"])
    if isinstance(r0, (list, tuple)) and r0:
        return int(r0[0])
    try:
        return int(list(r0.values())[0])  # type: ignore
    except Exception:
        return 0


def _mg_iter_stmt_pages(ing: MemgraphIngestor, limit: int) -> Iterable[List[Dict[str, Any]]]:
    # 以 id 排序分頁；同時帶出所屬 Function 與 File，便於 payload
    total = _mg_count_stmts(ing)
    if total == 0:
        yield []
        return
    pages = math.ceil(total / limit)
    for i in range(pages):
        skip = i * limit
        rows = ing.fetch_all(
            """
            MATCH (s:Stmt)<-[:CONTAINS_STMT]-(bb:BasicBlock)<-[:CONTAINS_BLOCK]-(f:Function)-[:DEFINED_IN]->(file:File)
            RETURN s.id AS sid, s.src AS src, s.span_start AS s, s.span_end AS e,
                   f.qualified_name AS qn, file.path AS path
            ORDER BY sid
            SKIP $skip LIMIT $limit
            """,
            {"skip": skip, "limit": limit},
        )
        out: List[Dict[str, Any]] = []
        for r in rows or []:
            if isinstance(r, dict):
                out.append({
                    "sid": r.get("sid"),
                    "src": r.get("src"),
                    "s": r.get("s"),
                    "e": r.get("e"),
                    "qn": r.get("qn"),
                    "path": r.get("path"),
                })
            else:
                try:
                    sid, src, s, e, qn, path = r
                    out.append({"sid": sid, "src": src, "s": s, "e": e, "qn": qn, "path": path})
                except Exception:
                    continue
        yield out


@app.command("push-stmts-to-qdrant")
def push_stmts_to_qdrant(
    repo_path: str | None = typer.Option(
        None, "--repo-path", help="Path to the repository (used for repo name in payload)"
    ),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of stmt records to push (debug)"),
) -> None:
    """
    將 Memgraph 內的 Stmt（含 src 與所屬 Function/File 資訊）嵌入並寫入 Qdrant。
    - 外鍵：payload 內含 sid（即 Stmt.id），可回跳 Memgraph 精準定位。
    - 若 s.src 缺失，可回退用 path+span 行號重建文字（通常不需要）。
    """
    if QdrantClient is None:
        console.print("[bold red]qdrant-client is not installed.[/bold red]")
        raise typer.Exit(1)

    # ---- Env/Settings
    repo_path = repo_path or settings.TARGET_REPO_PATH
    repo_root = Path(repo_path).resolve()
    mg_host = settings.MEMGRAPH_HOST
    mg_port = settings.MEMGRAPH_PORT

    q_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
    q_api = os.getenv("QDRANT_API_KEY", "")
    q_collection = os.getenv("QDRANT_STMT_COLLECTION", os.getenv("QDRANT_COLLECTION", "code_nodes"))
    q_distance = os.getenv("QDRANT_DISTANCE", "Cosine")
    q_batch = int(os.getenv("QDRANT_BATCH", "128"))
    q_timeout = float(os.getenv("QDRANT_TIMEOUT", "60"))
    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    resume_path = os.getenv("QDRANT_STMT_RESUME_STATE", ".tmp/qdrant_stmt_resume.json")

    emb_backend = os.getenv("EMBEDDING_BACKEND", "SENTENCE_TRANSFORMERS")
    emb_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    file_reader = FileReader(project_root=str(repo_root))

    # ---- Init embedder and qdrant
    embedder = _Embedder(emb_backend, emb_model)
    dim = len(embedder.embed(["dim-probe"])[0])

    qdr = QdrantClient(
        url=q_url,
        api_key=q_api or None,
        timeout=q_timeout,
        prefer_grpc=prefer_grpc,
    )
    _qdrant_ensure_collection(qdr, q_collection, dim, q_distance)

    # ---- Resume helpers
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)

    def load_resume():
        try:
            with open(resume_path, "r") as f:
                return json.load(f)
        except Exception:
            return {"page": 0, "offset_in_page": 0}

    def save_resume(state):
        tmp = resume_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, resume_path)

    state = load_resume()
    start_page = int(state.get("page", 0))
    start_offset = int(state.get("offset_in_page", 0))

    # ---- upsert with retry (wait=False)
    def upsert_with_retry(points: List[PointStruct], max_retries=3):
        backoff = 1.0
        for attempt in range(max_retries + 1):
            try:
                qdr.upsert(collection_name=q_collection, points=points, wait=False)
                return True
            except Exception as e:
                if attempt == max_retries:
                    console.print(f"[red]Upsert failed after retries: {e}[/red]")
                    return False
                time.sleep(backoff)
                backoff *= 2

    sent = 0
    processed = 0
    t0 = time.time()

    def _read_by_span(path: str, s: Any, e: Any) -> str:
        # span 為 (row,col) 雙組；這裡只用 row→行號（0-based → 1-based）
        if not path or s is None or e is None:
            return ""
        try:
            s_row = int(s[0]) + 1
            e_row = int(e[0]) + 1
            return file_reader.read(path, start_line=s_row, end_line=e_row)
        except Exception:
            return ""

    with MemgraphIngestor(host=mg_host, port=mg_port, batch_size=1000) as ing:
        total = _mg_count_stmts(ing)
        if limit and limit > 0:
            total = min(total, limit)
        console.print(f"[bold cyan]Preparing to push ~{total} Stmt nodes to Qdrant...[/bold cyan]")

        for page_index, page in enumerate(_mg_iter_stmt_pages(ing, limit=2000)):
            if page_index < start_page:
                continue
            if not page:
                save_resume({"page": page_index + 1, "offset_in_page": 0})
                continue
            if limit and processed >= limit:
                break

            texts: List[str] = []
            metas: List[Dict[str, Any]] = []

            inpage_start = start_offset if page_index == start_page else 0
            sent_this_page = 0

            for idx_in_page, r in enumerate(page):
                if idx_in_page < inpage_start:
                    continue
                if limit and processed >= limit:
                    break

                sid = str(r.get("sid") or "")
                src = r.get("src") or ""
                qn = str(r.get("qn") or "")
                path = str(r.get("path") or "")
                s_span = r.get("s")
                e_span = r.get("e")

                if not src:
                    # 回退：用檔案與行號重建
                    src = _read_by_span(path, s_span, e_span)
                if not sid or not src.strip():
                    save_resume({"page": page_index, "offset_in_page": idx_in_page + 1})
                    processed += 1
                    continue

                texts.append(src)
                metas.append({
                    "repo": repo_root.name,
                    "language": _guess_lang_by_ext(path),
                    "path": path,
                    "qualified_name": qn,
                    "sid": sid,
                    "kind": "stmt",
                    "span_start": s_span,
                    "span_end": e_span,
                })

                save_resume({"page": page_index, "offset_in_page": idx_in_page + 1})
                processed += 1

            if not texts:
                save_resume({"page": page_index + 1, "offset_in_page": 0})
                continue

            vecs = embedder.embed(texts)
            sub = max(1, q_batch)
            points: List[PointStruct] = []

            for payload, vec, text in zip(metas, vecs, texts):
                # 以 Stmt.id 為主要鍵（可復現），若超長可用哈希
                pid = int(hashlib.sha256(payload["sid"].encode()).hexdigest()[:16], 16)
                points.append(PointStruct(id=pid, vector=vec, payload={**payload, "code": text}))

            for i in range(0, len(points), sub):
                chunk = points[i:i+sub]
                ok = upsert_with_retry(chunk, max_retries=3)
                if ok:
                    sent += len(chunk)
                    sent_this_page += len(chunk)

            console.print(f"[green]Page {page_index}: upserted {sent_this_page} stmt-points (total {sent})[/green]")
            save_resume({"page": page_index + 1, "offset_in_page": 0})

    t1 = time.time()
    console.print(f"[bold green]Done. Upserted {sent} stmt points in {t1 - t0:.1f}s[/bold green]")




if __name__ == "__main__":
    app()
