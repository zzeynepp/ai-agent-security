# AI Agent Security Research

This repository contains hands-on experiments focused on the security of AI agents, tool-using LLM systems, and agentic workflows.

The goal is to understand how agent behavior changes when models reason across multiple steps, interact with tools, process untrusted observations, and encounter control boundaries.

## Projects

### 01 — Inside an AI Agent's Reasoning Loop

A small local ReAct agent used to explore:

- reasoning loops
- tool selection
- unnecessary tool calls
- termination failures
- `MAX_STEPS` circuit breakers
- duplicate action detection
- grounded fallback behavior
- untrusted tool output
- indirect prompt injection
- prompt-level vs. code-level controls
- security and utility trade-offs

[Read the lab and write-up](01-reasoning-loop/README.md)

## Repository Structure

```text
ai-agent-security/
├── README.md
└── 01-reasoning-loop/
    ├── README.md
    ├── agent.py
    ├── agent-vulnerable.py
    ├── llm.py
    ├── tools.py
    ├── test_agent.py
    ├── requirements.txt
    ├── documents/
    └── images/
