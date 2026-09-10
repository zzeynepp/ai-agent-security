import os

DOCS_DIR = os.path.join(os.path.dirname(__file__), "documents")


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

def read_document(name: str) -> str:
    """Read a file from the local documents directory."""
    name = name.strip().strip('"').strip("'")
    safe_name = os.path.basename(name)
    path = os.path.join(DOCS_DIR, safe_name)

    if not os.path.exists(path):
        available = ", ".join(sorted(os.listdir(DOCS_DIR)))
        return f"File '{safe_name}' was not found. Available files: {available}"

    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def lookup_policy(query: str) -> str:
    """Search the small local knowledge base."""

    query = query.strip().strip('"').strip("'").lower()

    matches = [
        value
        for key, value in KNOWLEDGE_BASE.items()
        if key in query or query in key
    ]

    if matches:
        return " ".join(matches)

    topics = ", ".join(KNOWLEDGE_BASE.keys())

    return (
        f"No entry found for '{query}'. "
        f"Available topics: {topics}"
    )


TOOLS = {
    "read_document": (
        read_document,
        "Read a file from the documents directory. "
        "Input: file name.",
    ),
    "lookup_policy": (
        lookup_policy,
        "Search the internal knowledge base. "
        "Input: topic or keyword.",
    ),
}


def run_tool(name: str, tool_input: str) -> str:
    name = name.strip()

    if name not in TOOLS:
        available = ", ".join(TOOLS.keys())

        return (
            f"Unknown tool '{name}'. "
            f"Available tools: {available}"
        )

    func, _ = TOOLS[name]

    try:
        return func(tool_input)
    except Exception as exc:
        return f"Tool error: {exc}"


def tools_description() -> str:
    lines = []

    for name, (_, description) in TOOLS.items():
        lines.append(f"- {name}: {description}")

    return "\n".join(lines)
