from pathlib import Path

from tree_sitter import Node, Parser
from typing import Optional, Iterable

def preview_code(file_reader, path: str, start: Optional[int], end: Optional[int], max_lines: int = 120) -> str:
    """
    Read a code span using the project's FileReader and return a numbered, truncated preview.
    - file_reader: an object exposing read(path, start_line=?, end_line=?) -> str
    - path: absolute file path in the repository
    - start, end: 1-based line numbers (inclusive). If any is falsy/invalid, returns "".
    - max_lines: safety limit
    """
    try:
        if not path or not start or not end or end < start:
            return ""
        text = file_reader.read(path, start_line=int(start), end_line=int(end))
        if not text:
            return ""
        lines = text.splitlines()
        shown = lines[:max_lines]
        suffix = "" if len(lines) <= max_lines else f"\\n... (truncated, showing {max_lines}/{len(lines)} lines)"
        numbered = [f"{i+int(start):>6}  {ln}" for i, ln in enumerate(shown)]
        return "\\n".join(numbered) + suffix
    except Exception:
        return ""


def build_c_qualified_name(node: Node, module_qn: str, name: str) -> str:
    """Build qualified name for C entities, handling namespaces properly."""
    module_parts = module_qn.split(".")

    is_module_file = (
        len(module_parts) >= 3  # At least project.dir.filename
        and (
            "interfaces" in module_parts
            or "modules" in module_parts
            or any(part.endswith((".h", ".c")) for part in module_parts)
        )
    )

    if is_module_file:
        project_name = module_parts[0]
        filename = module_parts[-1]
        return f"{project_name}.{filename}.{name}"
    else:
        path_parts = []
        current = node.parent

        while current and current.type != "translation_unit":
            if current.type == "namespace_definition":
                namespace_name = None
                name_node = current.child_by_field_name("name")
                if name_node and name_node.text:
                    namespace_name = name_node.text.decode("utf8")
                else:
                    for child in current.children:
                        if (
                            child.type in ["namespace_identifier", "identifier"]
                            and child.text
                        ):
                            namespace_name = child.text.decode("utf8")
                            break
                if namespace_name:
                    path_parts.append(namespace_name)
            current = current.parent

        path_parts.reverse()

        if path_parts:
            return f"{module_qn}.{'.'.join(path_parts)}.{name}"
        else:
            return f"{module_qn}.{name}"


def _extract_name_from_function_definition(func_node: Node) -> str | None:
    for child in func_node.children:
        if child.type == "function_declarator":
            return extract_c_function_name(child)
    return None


def _extract_name_from_declaration(func_node: Node) -> str | None:
    for child in func_node.children:
        if child.type == "function_declarator":
            return extract_c_function_name(child)
    return None


def _extract_name_from_function_declarator(func_node: Node) -> str | None:
    for child in func_node.children:
        if child.type in ["identifier"] and child.text:
            return child.text.decode("utf8") if child.text else None
    return None


def extract_c_function_name(func_node: Node) -> str | None:
    if func_node.type in [
        "function_definition",
    ]:
        return _extract_name_from_function_definition(func_node)

    elif func_node.type in [
        "declaration",
    ]:
        return _extract_name_from_declaration(func_node)

    elif func_node.type == "function_declarator":
        return _extract_name_from_function_declarator(func_node)

    return None


# Suffixes that indicate C++ source files
CPP_SOURCE_SUFFIXES = {
    ".cpp",
    ".hpp",
    ".cc",
    ".cxx",
    ".hxx",
    ".hh",
    ".ixx",
    ".cppm",
    ".ccm",
}

# The set of node.types in tree-sitter-cpp grammar that unambiguously indicate C++-only semantics
CPP_ONLY_NODE_TYPES = {
    "abstract_declarator",
    "access_specifier",  # public: private: etc.
    "alias_declaration",  # using T = int;
    "alignas_specifier",  # C++11 alignas
    "attribute_declaration",  # [[nodiscard]] etc.
    "auto",  # auto type deduction
    "cast_expression",  # static_cast / dynamic_cast / etc.
    "class_specifier",  # class definitions
    "concept_definition",  # C++20 concept
    "condition_clause",  # C++ specific if/switch condition forms
    "decltype_specifier",  # decltype(x)
    "delete_expression",  # delete ptr
    "dependent_type_specifier",  # dependent types in templates
    "enum_class_specifier",  # enum class
    "explicit_specifier",  # explicit keyword
    "friend_declaration",  # friend class X;
    "lambda_expression",  # lambdas
    "namespace_definition",  # namespace N {}
    "new_expression",  # new T()
    "noexcept_specifier",  # noexcept
    "operator_cast",  # conversion operators
    "override_specifier",  # override keyword
    "parameter_pack_expansion",  # T... variadic
    "qualified_identifier",  # std::vector etc.
    "reference_declarator",  # T& or T&&
    "static_assert_declaration",  # static_assert(...)
    "template_argument_list",  # std::vector<int> etc.
    "template_declaration",  # template<typename T>
    "template_function",  # function templates
    "template_method",  # methods in template classes
    "template_parameter_list",
    "template_type",
    "this",  # this pointer
    "throw_specifier",  # throw() specifiers (older C++)
    "type_qualifier",  # constexpr, consteval, constinit etc.
    "using_declaration",  # using namespace etc.
    "virtual_function_specifier",  # virtual keyword
    "virtual_specifier",  # final, etc.
}


def determine_if_cpp_header(file_path: Path, cpp_parser: Parser) -> bool:
    """
    Determines if a .h file should be treated as a C++ header.

    Returns True if it's likely a C++ header (has C++-only AST constructs
    or there are sibling C++ source files), otherwise False (treat as C header).
    """

    # 1. Sibling check: same directory, any file with C++ suffix
    parent = file_path.parent
    try:
        for sibling in parent.iterdir():
            if sibling == file_path:
                continue
            if sibling.is_file():
                if sibling.suffix.lower() in CPP_SOURCE_SUFFIXES:
                    return True
    except Exception:
        # If directory listing fails, ignore sibling heuristic
        pass

    # 2. Parse with tree-sitter-cpp parser
    try:
        source_bytes = file_path.read_bytes()
        # Alternatively: read text and encode utf-8
    except Exception:
        # If can't read file, conservatively treat as C header
        return False

    # If file is empty or nearly so, no strong indication of C++
    if not source_bytes.strip():
        return False

    tree = cpp_parser.parse(source_bytes)
    root: Node = tree.root_node

    # Recursive search for any node whose type is in CPP_ONLY_NODE_TYPES
    def has_cpp_type(node: Node) -> bool:
        if node.type in CPP_ONLY_NODE_TYPES:
            return True
        # Recurse into named children (faster / more relevant than all children)
        for child in node.named_children:
            # Early exit
            if has_cpp_type(child):
                return True
        return False

    if has_cpp_type(root):
        return True

    # If no C++-only node types found, treat as C header
    return False
