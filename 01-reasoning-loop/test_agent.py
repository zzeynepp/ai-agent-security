import llm
import agent
from fastapi.testclient import TestClient


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0

    def __call__(self, prompt, stop=None, temperature=0.2):
        response = self.responses[
            min(self.index, len(self.responses) - 1)
        ]
        self.index += 1
        return response


TEST_MESSAGES = {
    "direct answer": "What can you help me with?",
    "policy lookup": "What is the account recovery policy?",
    "repeated action": "What is the account recovery policy?",
    "format retry": "What is the account recovery policy?",
    "document read": "Please summarize ticket_normal.txt",
}


def run_case(name, responses):
    print("\n" + "=" * 60)
    print(f"TEST: {name}")
    print("=" * 60)

    llm.generate = ScriptedLLM(responses)
    agent.SESSIONS.clear()

    answer = agent.react_loop(
        "test-session",
        TEST_MESSAGES[name],
    )

    print("\nresult:", repr(answer))
    return answer


# No tool needed
answer = run_case(
    "direct answer",
    [
        "Thought: This is a general question and does not require a tool.\n"
        "Final Answer: I can help with common IT issues, account access, "
        "software questions, and internal IT policies."
    ],
)

assert "IT issues" in answer


# Tool call followed by a final answer
answer = run_case(
    "policy lookup",
    [
        "Thought: I should check the account recovery policy.\n"
        "Action: lookup_policy\n"
        "Action Input: account recovery",

        "Thought: The policy information is enough to answer.\n"
        "Final Answer: Account recovery requires identity verification "
        "before the reset process can be completed.",
    ],
)

assert "identity verification" in answer


# Same tool call twice should stop the loop
answer = run_case(
    "repeated action",
    [
        "Thought: I should check the policy.\n"
        "Action: lookup_policy\n"
        "Action Input: account recovery",

        "Thought: I will check the same policy again.\n"
        "Action: lookup_policy\n"
        "Action Input: account recovery",

        "Account recovery requires identity verification before "
        "the reset process can be completed.",
    ],
)

assert "identity verification" in answer


# Invalid output should trigger a retry and then fallback
answer = run_case(
    "format retry",
    [
        "Thought: I should check the policy.\n"
        "Action: lookup_policy\n"
        "Action Input: account recovery",

        "I think I have enough information.",

        "This still does not match the expected format.",

        "Account recovery requires identity verification before "
        "the reset process can be completed.",
    ],
)

assert not answer.lower().startswith("thought:")
assert "identity verification" in answer


# Read a normal local document
answer = run_case(
    "document read",
    [
        "Thought: I need to read the ticket before summarizing it.\n"
        "Action: read_document\n"
        "Action Input: ticket_normal.txt",

        "Thought: I have enough information from the document.\n"
        "Final Answer: The ticket describes a printer issue where "
        "documents remain stuck in the print queue.",
    ],
)

assert "printer" in answer.lower()


print("\n" + "=" * 60)
print("TEST: API endpoints")
print("=" * 60)

client = TestClient(agent.app)

health = client.get("/health").json()
print("/health ->", health)

assert health["status"] == "healthy"
assert health["agent"] == "Local Helpdesk Agent"


llm.generate = ScriptedLLM(
    [
        "Thought: This is a simple greeting.\n"
        "Final Answer: Hello! How can I help?"
    ]
)

response = client.post(
    "/chat",
    json={"message": "Hello"},
).json()

print("/chat ->", response)

assert "session_id" in response
assert response["response"]


print("\nAll tests passed.")
