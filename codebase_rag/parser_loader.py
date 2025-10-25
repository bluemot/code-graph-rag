import importlib
import os
import subprocess
import sys
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from loguru import logger
from tree_sitter import Language, Parser, Query

from .language_config import LANGUAGE_CONFIGS

# Define a type for the language library loaders
LanguageLoader = Callable[[], object] | None


def _try_load_from_submodule(lang_name: str, vendor=False) -> LanguageLoader:
    """Try to load language from git submodule Python bindings."""
    if vendor:
        vendor_path = os.path.join("grammars", "vendor")
        submodule_path = os.path.join(vendor_path, f"tree-sitter-{lang_name}")
    else:
        submodule_path = os.path.join("grammars", f"tree-sitter-{lang_name}")
    python_bindings_path = os.path.join(submodule_path, "bindings", "python")
    logger.debug(f"{submodule_path}:{python_bindings_path}")
    if not os.path.exists(python_bindings_path):
        return None
    logger.debug(f"{submodule_path}:{python_bindings_path}=>exist")
    try:
        # Add the Python bindings to path
        if python_bindings_path not in sys.path:
            sys.path.insert(0, python_bindings_path)

        try:
            # Check if we need to build the binding
            module_name = f"tree_sitter_{lang_name.replace('-', '_')}"

            # Try to build and install the Python binding
            setup_py = os.path.join(submodule_path, "setup.py")
            if os.path.exists(setup_py):
                logger.debug(f"Building Python bindings for {lang_name}...")
                result = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=submodule_path,
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    logger.debug(
                        f"Failed to build {lang_name} bindings: stdout={result.stdout}, stderr={result.stderr}"
                    )
                    return None
                logger.debug(f"Successfully built {lang_name} bindings")

            # Now try to import the module
            logger.debug(f"Attempting to import module: {module_name}")
            module = importlib.import_module(module_name)

            # Try different possible language attribute names
            language_attrs = [
                "language",
                f"language_{lang_name}",
                f"language_{lang_name.replace('-', '_')}",
            ]

            for attr_name in language_attrs:
                if hasattr(module, attr_name):
                    logger.debug(
                        f"Successfully loaded {lang_name} from submodule bindings using {attr_name}"
                    )
                    return getattr(module, attr_name)  # type: ignore[no-any-return]

            logger.debug(
                f"Module {module_name} imported but has no language attribute. Available: {dir(module)}"
            )

        finally:
            # Clean up path
            if python_bindings_path in sys.path:
                sys.path.remove(python_bindings_path)

    except Exception as e:
        logger.debug(f"Failed to load {lang_name} from submodule bindings: {e}")

    return None


# Import available Tree-sitter languages and correctly type them as Optional
def _import_language_loaders() -> dict[str, LanguageLoader]:
    """Import language loaders with proper error handling and typing."""
    loaders: dict[str, LanguageLoader] = {}
    try:
        from tree_sitter_c import language as c_language_so
        loaders["c"] = c_language_so
        logger.debug("load existed c lib")
    except ImportError:
        loaders["c"] = _try_load_from_submodule("c", True) #use tree-sitter-c from original github

    try:
        from tree_sitter_python import language as python_language_so

        loaders["python"] = python_language_so
    except ImportError:
        loaders["python"] = _try_load_from_submodule("python")

    try:
        from tree_sitter_javascript import language as javascript_language_so

        loaders["javascript"] = javascript_language_so
    except ImportError:
        loaders["javascript"] = _try_load_from_submodule("javascript")

    try:
        from tree_sitter_typescript import language_typescript as typescript_language_so

        loaders["typescript"] = typescript_language_so
    except ImportError:
        loaders["typescript"] = _try_load_from_submodule("typescript")

    try:
        from tree_sitter_rust import language as rust_language_so

        loaders["rust"] = rust_language_so
    except ImportError:
        loaders["rust"] = _try_load_from_submodule("rust")

    try:
        from tree_sitter_go import language as go_language_so

        loaders["go"] = go_language_so
    except ImportError:
        loaders["go"] = _try_load_from_submodule("go")

    try:
        from tree_sitter_scala import language as scala_language_so

        loaders["scala"] = scala_language_so
    except ImportError:
        loaders["scala"] = _try_load_from_submodule("scala")

    try:
        from tree_sitter_java import language as java_language_so

        loaders["java"] = java_language_so
    except ImportError:
        loaders["java"] = _try_load_from_submodule("java")

    try:
        from tree_sitter_cpp import language as cpp_language_so

        loaders["cpp"] = cpp_language_so
    except ImportError:
        loaders["cpp"] = _try_load_from_submodule("cpp")

    # Lua support
    try:
        from tree_sitter_lua import language as lua_language_so

        loaders["lua"] = lua_language_so
    except ImportError:
        loaders["lua"] = _try_load_from_submodule("lua")

    # Automatically try submodule loading for any language not already loaded
    for lang_name in LANGUAGE_CONFIGS.keys():
        if lang_name not in loaders or loaders[lang_name] is None:
            loaders[lang_name] = _try_load_from_submodule(lang_name)

    return loaders


_language_loaders = _import_language_loaders()


LANGUAGE_LIBRARIES: dict[str, LanguageLoader | None] = _language_loaders


def load_parsers() -> tuple[dict[str, Parser], dict[str, Any]]:
    """Loads all available Tree-sitter parsers and compiles their queries."""
    parsers: dict[str, Parser] = {}
    queries: dict[str, Any] = {}
    available_languages = []

    # Deepcopy to avoid modifying the original configs
    configs = deepcopy(LANGUAGE_CONFIGS)
    # breakpoint()
    for lang_name, lang_config in configs.items():
        if lang_lib := LANGUAGE_LIBRARIES.get(lang_name):
            try:
                language = Language(lang_lib())
                parser = Parser(language)

                parsers[lang_name] = parser

                # Create Tree-sitter queries - use pre-formatted queries if available, otherwise generate from node types
                function_patterns = (
                    lang_config.function_query
                    if lang_config.function_query
                    else " ".join(
                        [
                            f"({node_type}) @function"
                            for node_type in lang_config.function_node_types
                        ]
                    )
                )
                class_patterns = (
                    lang_config.class_query
                    if lang_config.class_query
                    else " ".join(
                        [
                            f"({node_type}) @class"
                            for node_type in lang_config.class_node_types
                        ]
                    )
                )
                call_patterns = (
                    lang_config.call_query
                    if lang_config.call_query
                    else " ".join(
                        [
                            f"({node_type}) @call"
                            for node_type in lang_config.call_node_types
                        ]
                    )
                )

                # Create import query patterns
                import_patterns = " ".join(
                    [
                        f"({node_type}) @import"
                        for node_type in lang_config.import_node_types
                    ]
                )
                import_from_patterns = " ".join(
                    [
                        f"({node_type}) @import_from"
                        for node_type in lang_config.import_from_node_types
                    ]
                )

                # Combine import patterns (remove duplicates)
                all_import_patterns = []
                if import_patterns.strip():
                    all_import_patterns.append(import_patterns)
                if (
                    import_from_patterns.strip()
                    and import_from_patterns != import_patterns
                ):
                    all_import_patterns.append(import_from_patterns)
                combined_import_patterns = " ".join(all_import_patterns)

                # Create locals query for variable tracking
                locals_query = None
                if lang_name in ("javascript", "typescript"):
                    # Create language-specific locals patterns
                    if lang_name == "javascript":
                        locals_patterns = """
                        ; Variable definitions
                        (variable_declarator name: (identifier) @local.definition)
                        (function_declaration name: (identifier) @local.definition)
                        (class_declaration name: (identifier) @local.definition)

                        ; Variable references
                        (identifier) @local.reference
                        """
                    else:  # typescript
                        # TypeScript-specific patterns (comprehensive like JavaScript)
                        locals_patterns = """
                        ; Variable definitions (TypeScript has multiple declaration types)
                        (variable_declarator name: (identifier) @local.definition)
                        (lexical_declaration (variable_declarator name: (identifier) @local.definition))
                        (variable_declaration (variable_declarator name: (identifier) @local.definition))

                        ; Function definitions
                        (function_declaration name: (identifier) @local.definition)

                        ; Class definitions (uses type_identifier for class names)
                        (class_declaration name: (type_identifier) @local.definition)

                        ; Variable references
                        (identifier) @local.reference
                        """

                    try:
                        locals_query = Query(language, locals_patterns)
                    except Exception as e:
                        logger.debug(
                            f"Failed to create locals query for {lang_name}: {e}"
                        )
                        locals_query = None

                queries[lang_name] = {
                    "functions": (
                        Query(language, function_patterns)
                        if function_patterns
                        else None
                    ),
                    "classes": (
                        Query(language, class_patterns) if class_patterns else None
                    ),
                    "calls": Query(language, call_patterns) if call_patterns else None,
                    "imports": (
                        Query(language, combined_import_patterns)
                        if combined_import_patterns
                        else None
                    ),
                    "locals": locals_query,
                    "config": lang_config,
                    "language": language,
                }

                available_languages.append(lang_name)
                logger.success(f"Successfully loaded {lang_name} grammar.")
            except Exception as e:
                logger.warning(f"Failed to load {lang_name} grammar: {e}")
        else:
            logger.debug(f"Tree-sitter library for {lang_name} not available.")

    if not available_languages:
        raise RuntimeError("No Tree-sitter languages available.")

    logger.info(f"Initialized parsers for: {', '.join(available_languages)}")
    return parsers, queries
