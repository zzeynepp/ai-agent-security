"""Local tools for the reasoning-loop lab."""

import os
import traceback
from dataclasses import dataclass

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")

MAX_TOOL_OUTPUT_CHARS = 4000
MIN_POLICY_QUERY_LEN = 3


@dataclass
class ToolResult:
    ok: bool
    data: str = ""              # untrusted tool output
    error: str = ""             # orchestrator message
    normalized_input: str = ""  # input as the tool resolved it
    truncated: bool = False


KNOWLEDGE_BASE = {
    "account recovery": (
        "Account recovery requires identity verification before "
        "the reset process can be completed."
    ),
    "software install": (
        "Software installation requests must be submitted through "
        "the internal service portal."
    ),
    "wifi": (
        "Corporate Wi-Fi access is available only to managed devices."
    ),
    "printer": (
        "If a print job is stuck, clear the queue and retry. "
        "If the issue continues, reinstall the printer."
    ),
}


def _strip_quotes(raw: str) -> str:
    return raw.strip().strip('"').strip("'").strip()


def normalize_document_name(raw: str) -> str:
    """Normalize a document name the way read_document will use it."""
    return os.path.basename(_strip_quotes(raw))


def normalize_policy_query(raw: str) -> str:
    """Normalize a policy query the way lookup_policy will use it."""
    return _strip_quotes(raw).lower()


def read_document(name: str) -> ToolResult:
    """Read a file from the local documents directory."""
    safe_name = normalize_document_name(name)
    path = os.path.join(DOCS_DIR, safe_name)

    if not safe_name or not os.path.isfile(path):
        available = ", ".join(sorted(os.listdir(DOCS_DIR)))

        return ToolResult(
            ok=False,
            error=(
                f"File '{safe_name}' was not found. "
                f"Available files: {available}"
            ),
            normalized_input=safe_name,
        )

    with open(path, "r", encoding="utf-8") as file:
        content = file.read(MAX_TOOL_OUTPUT_CHARS + 1)

    truncated = len(content) > MAX_TOOL_OUTPUT_CHARS

    return ToolResult(
        ok=True,
        data=content[:MAX_TOOL_OUTPUT_CHARS],
        normalized_input=safe_name,
        truncated=truncated,
    )


def lookup_policy(query: str) -> ToolResult:
    """Search the small local knowledge base."""
    normalized = normalize_policy_query(query)
    topics = ", ".join(KNOWLEDGE_BASE.keys())

    if len(normalized) < MIN_POLICY_QUERY_LEN:
        return ToolResult(
            ok=False,
            error=(
                f"Query '{normalized}' is too short to match a topic. "
                f"Available topics: {topics}"
            ),
            normalized_input=normalized,
        )

    matches = [
        value
        for key, value in KNOWLEDGE_BASE.items()
        if key in normalized or normalized in key
    ]

    if not matches:
        return ToolResult(
            ok=False,
            error=(
                f"No entry found for '{normalized}'. "
                f"Available topics: {topics}"
            ),
            normalized_input=normalized,
        )

    return ToolResult(
        ok=True,
        data=" ".join(matches),
        normalized_input=normalized,
    )


TOOLS = {
    "read_document": (
        read_document,
        normalize_document_name,
        "Read a file from the documents directory. Input: file name.",
    ),
    "lookup_policy": (
        lookup_policy,
        normalize_policy_query,
        "Search the internal knowledge base. Input: topic or keyword.",
    ),
}


def normalize_action(name: str) -> str:
    """Normalize a tool name the way run_tool will resolve it."""
    return name.strip().lower()


def normalize_input(name: str, tool_input: str) -> str:
    """Normalize a tool input the way the named tool will use it."""
    entry = TOOLS.get(normalize_action(name))

    if entry is None:
        return _strip_quotes(tool_input)

    _, normalizer, _ = entry

    return normalizer(tool_input)


def run_tool(name: str, tool_input: str) -> ToolResult:
    action = normalize_action(name)

    if action not in TOOLS:
        available = ", ".join(TOOLS.keys())

        return ToolResult(
            ok=False,
            error=(
                f"Unknown tool '{name.strip()}'. "
                f"Available tools: {available}"
            ),
            normalized_input=_strip_quotes(tool_input),
        )

    func, _, _ = TOOLS[action]

    try:
        return func(tool_input)
    except Exception:
        # keep exception text out of the model context
        traceback.print_exc()

        return ToolResult(
            ok=False,
            error="The tool could not complete the request.",
            normalized_input=normalize_input(action, tool_input),
        )


def tools_description() -> str:
    lines = []

    for name, (_, _, description) in TOOLS.items():
        lines.append(f"- {name}: {description}")

    return "\n".join(lines)
