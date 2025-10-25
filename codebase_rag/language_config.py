from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


# Shared node type constants to eliminate duplication
COMMON_JS_TS_FUNCTIONS = [
    "function_declaration",
    "generator_function_declaration",
    "function_expression",
    "arrow_function",
    "method_definition",
]

COMMON_JS_TS_CLASSES = ["class_declaration", "class"]

COMMON_JS_TS_IMPORTS = [
    "import_statement",
    "lexical_declaration",
    "export_statement",
]

COMMON_DECLARATION_IMPORT = ["import_declaration"]

COMMON_USING_DIRECTIVE = ["using_directive"]

CPP_IMPORTS = [
    "preproc_include",
    "template_function",
    "declaration",
]  # #include, import <>, module declarations


def create_lang_config(**kwargs: Any) -> "LanguageConfig":
    """Helper to create LanguageConfig without redundant name assignment."""
    # Name will be set automatically when configs are processed
    return LanguageConfig(name="", **kwargs)


@dataclass
class LanguageConfig:
    """Configuration for language-specific Tree-sitter parsing."""

    name: str
    file_extensions: list[str]

    # AST node type mappings to semantic concepts
    function_node_types: list[str]
    class_node_types: list[str]
    module_node_types: list[str]
    call_node_types: list[str] = field(default_factory=list)

    # Import statement node types for precise import resolution
    import_node_types: list[str] = field(default_factory=list)
    import_from_node_types: list[str] = field(default_factory=list)

    # Field names for extracting names
    name_field: str = "name"
    body_field: str = "body"

    # Package detection patterns
    package_indicators: list[str] = field(
        default_factory=list
    )  # e.g., ["__init__.py"] for Python

    # Optional pre-formatted Tree-sitter query strings or query generators
    # These override the default node_types-based query generation
    function_query: str | None = None
    class_query: str | None = None
    call_query: str | None = None


######################## Language configurations ###############################
# Automatic generation might add types that are too broad or inaccurate.
# You have to manually check and adjust the configurations after running the
# automatic generation.
################################################################################

LANGUAGE_CONFIGS = {
    "c": create_lang_config(
        file_extensions=[".c", ".h"],
        function_node_types=["function_definition"],
        class_node_types=["struct_specifier", "union_specifier", "enum_specifier"],
        module_node_types=[],
        call_node_types=["call_expression"],
        import_node_types=["preproc_include"],
        import_from_node_types=["preproc_include"],
        function_query="""
            (function_definition
               (function_declarator
                    (identifier) @name) ) @function
            """,
        call_query="""
            (call_expression
                (identifier) @name ) @call
            """,
    ),
    "python": create_lang_config(
        file_extensions=[".py"],
        function_node_types=["function_definition"],
        class_node_types=["class_definition"],
        module_node_types=["module"],
        call_node_types=["call", "with_statement"],
        import_node_types=["import_statement"],
        import_from_node_types=["import_from_statement"],
        package_indicators=["__init__.py"],
    ),
    "javascript": create_lang_config(
        file_extensions=[".js", ".jsx"],
        function_node_types=COMMON_JS_TS_FUNCTIONS,
        class_node_types=COMMON_JS_TS_CLASSES,
        module_node_types=["program"],
        call_node_types=["call_expression"],
        import_node_types=COMMON_JS_TS_IMPORTS,
        import_from_node_types=COMMON_JS_TS_IMPORTS,  # Include CommonJS require and re-exports
    ),
    "typescript": create_lang_config(
        file_extensions=[".ts", ".tsx"],
        function_node_types=COMMON_JS_TS_FUNCTIONS
        + ["function_signature"],  # For ambient declarations: declare function
        class_node_types=COMMON_JS_TS_CLASSES
        + [
            "abstract_class_declaration",
            "enum_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "internal_module",
        ],
        module_node_types=["program"],
        call_node_types=["call_expression"],
        import_node_types=COMMON_JS_TS_IMPORTS,
        import_from_node_types=COMMON_JS_TS_IMPORTS,  # Include CommonJS require and re-exports
    ),
    "rust": create_lang_config(
        file_extensions=[".rs"],
        function_node_types=[
            "function_item",  # Regular functions: fn name() {}
            "function_signature_item",  # Function signatures in traits
            "closure_expression",  # Closures/lambdas: |x| x + 1
        ],
        class_node_types=[
            "struct_item",  # Struct definitions
            "enum_item",  # Enum definitions
            "union_item",  # Union definitions
            "trait_item",  # Trait definitions
            "impl_item",  # Implementation blocks
            "type_item",  # Type aliases: type Name = Type;
        ],
        module_node_types=[
            "source_file",  # Root module file
            "mod_item",  # Module declarations: mod name {}
        ],
        call_node_types=[
            "call_expression",  # Function and method calls
            "macro_invocation",  # Macro calls: println!()
        ],
        import_node_types=["use_declaration", "extern_crate_declaration"],
        import_from_node_types=["use_declaration"],  # Rust uses 'use' for all imports
        package_indicators=["Cargo.toml"],  # Rust's package manifest
        # Pre-formatted Tree-sitter queries based on official tree-sitter-rust grammar
        function_query="""
        (function_item
            name: (identifier) @name) @function
        (function_signature_item
            name: (identifier) @name) @function
        (closure_expression) @function
        """,
        class_query="""
        (struct_item
            name: (type_identifier) @name) @class
        (enum_item
            name: (type_identifier) @name) @class
        (union_item
            name: (type_identifier) @name) @class
        (trait_item
            name: (type_identifier) @name) @class
        (type_item
            name: (type_identifier) @name) @class
        (impl_item) @class
        (mod_item
            name: (identifier) @name) @module
        """,
        call_query="""
        (call_expression
            function: (identifier) @name) @call
        (call_expression
            function: (field_expression
                field: (field_identifier) @name)) @call
        (call_expression
            function: (scoped_identifier
                "::"
                name: (identifier) @name)) @call
        (macro_invocation
            macro: (identifier) @name) @call
        """,
    ),
    "go": create_lang_config(
        file_extensions=[".go"],
        function_node_types=["function_declaration", "method_declaration"],
        class_node_types=["type_declaration"],  # Go structs
        module_node_types=["source_file"],
        call_node_types=["call_expression"],
        import_node_types=["import_declaration"],
        import_from_node_types=["import_declaration"],  # Go uses same node for imports
    ),
    "scala": create_lang_config(
        file_extensions=[".scala", ".sc"],
        function_node_types=[
            "function_definition",
            "function_declaration",
        ],
        class_node_types=[
            "class_definition",
            "object_definition",
            "trait_definition",
        ],
        module_node_types=["compilation_unit"],
        call_node_types=[
            "call_expression",
            "generic_function",
            "field_expression",
            "infix_expression",
        ],
        import_node_types=COMMON_DECLARATION_IMPORT,
        import_from_node_types=COMMON_DECLARATION_IMPORT,  # Scala uses same node for imports
        package_indicators=[],  # Scala uses package declarations
    ),
    "java": create_lang_config(
        file_extensions=[".java"],
        function_node_types=[
            "method_declaration",
            "constructor_declaration",
        ],
        class_node_types=[
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "annotation_type_declaration",
            "record_declaration",
        ],
        module_node_types=["program"],
        package_indicators=[],  # Java uses package declarations
        call_node_types=["method_invocation"],
        import_node_types=COMMON_DECLARATION_IMPORT,
        import_from_node_types=COMMON_DECLARATION_IMPORT,  # Java uses same node for imports
        # Pre-formatted Tree-sitter queries for comprehensive Java parsing
        function_query="""
        (method_declaration
            name: (identifier) @name) @function
        (constructor_declaration
            name: (identifier) @name) @function
        """,
        class_query="""
        (class_declaration
            name: (identifier) @name) @class
        (interface_declaration
            name: (identifier) @name) @class
        (enum_declaration
            name: (identifier) @name) @class
        (annotation_type_declaration
            name: (identifier) @name) @class
        (record_declaration
            name: (identifier) @name) @class
        """,
        call_query="""
        (method_invocation
            name: (identifier) @name) @call
        (object_creation_expression
            type: (type_identifier) @name) @call
        """,
    ),
    "cpp": create_lang_config(
        file_extensions=[
            ".cpp",
            ".h",
            ".hpp",
            ".cc",
            ".cxx",
            ".hxx",
            ".hh",
            ".ixx",
            ".cppm",
            ".ccm",
        ],
        function_node_types=[
            "function_definition",  # Includes aliased constructor/destructor/operator definitions
            "declaration",  # Includes aliased constructor/destructor/operator declarations
            "field_declaration",  # For method declarations in classes
            "template_declaration",  # For template functions
            "lambda_expression",  # For lambda functions
        ],
        class_node_types=[
            "class_specifier",
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
        ],
        module_node_types=[
            "translation_unit",
            "namespace_definition",
            "linkage_specification",  # extern "C" blocks
            "declaration",  # For module declarations like "module math_operations;"
        ],
        call_node_types=[
            "call_expression",
            "field_expression",  # For method calls like obj.method()
            "subscript_expression",  # For operator[] calls
            "new_expression",  # For new operator
            "delete_expression",  # For delete operator
            "binary_expression",  # For operator overloads like obj1 + obj2
            "unary_expression",  # For unary operators like ++obj
            "update_expression",  # For prefix/postfix increment/decrement
        ],
        import_node_types=CPP_IMPORTS,
        import_from_node_types=CPP_IMPORTS,
        # C++ specific configurations
        package_indicators=["CMakeLists.txt", "Makefile", "*.vcxproj", "conanfile.txt"],
        # Pre-formatted Tree-sitter queries for comprehensive C++ parsing
        function_query="""
    (function_definition) @function
    (template_declaration (function_definition)) @function
    (lambda_expression) @function
    (field_declaration) @function
    (declaration) @function
    """,
        class_query="""
    (class_specifier) @class
    (struct_specifier) @class
    (union_specifier) @class
    (enum_specifier) @class
    (template_declaration (class_specifier)) @class
    (template_declaration (struct_specifier)) @class
    (template_declaration (union_specifier)) @class
    (template_declaration (enum_specifier)) @class
    """,
        call_query="""
    (call_expression) @call
    (binary_expression) @call
    (unary_expression) @call
    (update_expression) @call
    (field_expression) @call
    (subscript_expression) @call
    (new_expression) @call
    (delete_expression) @call
    """,
    ),
    "c-sharp": create_lang_config(
        file_extensions=[".cs"],
        function_node_types=[
            "destructor_declaration",
            "local_function_statement",
            "function_pointer_type",
            "constructor_declaration",
            "anonymous_method_expression",
            "lambda_expression",
            "method_declaration",
        ],
        class_node_types=[
            "class_declaration",
            "struct_declaration",
            "enum_declaration",
            "interface_declaration",
        ],
        module_node_types=["compilation_unit"],
        call_node_types=["invocation_expression"],
        import_node_types=COMMON_USING_DIRECTIVE,
        import_from_node_types=COMMON_USING_DIRECTIVE,  # C# uses using directives
    ),
    "php": create_lang_config(
        file_extensions=[".php"],
        function_node_types=[
            "function_static_declaration",
            "anonymous_function",
            "function_definition",
            "arrow_function",
        ],
        class_node_types=[
            "trait_declaration",
            "enum_declaration",
            "interface_declaration",
            "class_declaration",
        ],
        module_node_types=["program"],
        call_node_types=[
            "member_call_expression",
            "scoped_call_expression",
            "function_call_expression",
            "nullsafe_member_call_expression",
        ],
    ),
    "lua": create_lang_config(
        file_extensions=[".lua"],
        function_node_types=[
            "function_declaration",
            "function_definition",  # For assignment patterns: Calculator.divide = function() end
        ],
        class_node_types=[],
        module_node_types=["chunk"],
        call_node_types=["function_call"],
        import_node_types=["function_call"],
    ),
}


def _initialize_config_names() -> None:
    """Initialize config names based on dict keys."""
    for lang_name, config in LANGUAGE_CONFIGS.items():
        if not config.name:  # Only set if empty (from create_lang_config)
            config.name = lang_name


# Initialize names on module load
_initialize_config_names()


def get_language_config(file_extension: str) -> LanguageConfig | None:
    """Get language configuration based on file extension."""
    for config in LANGUAGE_CONFIGS.values():
        if file_extension in config.file_extensions:
            return config
    return None


def get_language_config_by_name(language_name: str) -> LanguageConfig | None:
    """Get language configuration by language name."""
    return LANGUAGE_CONFIGS.get(language_name.lower())
