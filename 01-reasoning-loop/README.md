# Inside an AI Agent's Reasoning Loop

Reading about reasoning loops in diagrams wasn't enough for me. I wanted to see what actually happens when an agent starts making decisions, calling tools, processing observations, and deciding when to stop.

So I built a small ReAct-style helpdesk agent in my local environment and used it as a lab. The point wasn't to build a production agent. I wanted the loop to stay small enough that I could watch each transition and deliberately break parts of it.

At first, the flow looks straightforward:

```text
Reason → Action → Observation → Reason
```

Once I started running it, a few less-obvious behaviors showed up. The model sometimes used a tool when the task didn't need one. It could repeat an action instead of terminating. More importantly, data returned by a tool became part of the context used for the next decision.

That made the reasoning loop interesting to me not only as an agent design pattern, but also as a security boundary.

---

## 1. Chatbot vs. AI Agent

A basic chatbot can be represented like this:

```text
User
  ↓
LLM
  ↓
Response
```

An agent introduces more moving parts:

```text
User
  ↓
LLM
  ↓
Reason
  ↓
Action
  ↓
Tool
  ↓
Observation
  ↓
LLM
  ↓
Final Answer
```

The model is no longer only generating an answer. It can decide whether external information is needed and which available tool should be used.

One distinction matters here: **the LLM does not execute the tool itself.** It produces text describing the requested action. The surrounding orchestrator parses that output and executes the tool.

For example:

```text
Thought: I need information from the internal knowledge base.
Action: lookup_policy
Action Input: account recovery
```

The Python side then does the actual work:

```python
observation = tools.run_tool(action, tool_input)
```

The result is passed back to the model as an observation. That transition is where the loop starts to become interesting.

---

## 2. What Is a Reasoning Loop?

A reasoning loop lets an agent work through a task over multiple model calls instead of trying to solve everything in one response.

If the model needs external information:

```text
Reason
  ↓
Action
  ↓
Tool
  ↓
Observation
```

The observation is added to the context and the model gets another turn. Once enough information is available, the loop should terminate:

```text
Reason
  ↓
Final Answer
```

A compact way to think about it is:

**Reason → Act → Observe → Repeat**

The important part is that every observation can influence what happens on the next turn.

---

## 3. Building a Minimal ReAct Agent

The lab uses a local Ollama model and two deliberately small tools:

```text
lookup_policy()
read_document()
```

A policy question can produce a flow like this:

```text
User: What is the account recovery policy?

Thought: I should check the account recovery policy.
Action: lookup_policy
Action Input: account recovery
```

The orchestrator parses the action, executes the tool, and gets:

```text
Observation:
Account recovery requires identity verification before the reset process can be completed.
```

The model now has the original question plus information that did not exist in its first turn. It has to decide what to do next.

The core loop in `agent.py` intentionally stays simple. It parses `Action`, `Action Input`, and `Final Answer`, keeps a scratchpad of previous steps, executes allowlisted tools, and stops when the model returns a final answer or an orchestrator-level termination control fires.

---

## 4. Reason → Action → Observation → Reason

A normal run looks roughly like this:

```text
User
  ↓
Reason
  ↓
Action
  ↓
Tool
  ↓
Observation
  ↓
Reason
  ↓
Final Answer
```

One detail stood out to me: the tool result isn't necessarily returned directly to the user. It goes back into the model context first.

```text
Tool Output
    ↓
Observation
    ↓
Next Reasoning
    ↓
Next Decision
```

That is what makes an agent adaptive. It is also what creates a trust boundary: the next decision can only be as reliable as the context influencing it.

---

## 5. When the Model Should NOT Call a Tool

When reviewing tool selection, the obvious question is:

> Did the model choose the correct tool?

But there is an earlier question:

> **Should the model have used a tool at all?**

I saw this clearly when I asked the agent to summarize a local support ticket. I expected:

```text
read_document
→ Observation
→ Final Answer
```

Instead, the agent read the document and then performed a policy lookup before answering:

```text
read_document
→ Observation
→ lookup_policy
→ Observation
→ Final Answer
```

![Multi-step reasoning with an unnecessary tool call](images/01-multi-step-reasoning.png)

The second tool call wasn't necessary for the original task. The agent had quietly expanded the task from summarization into troubleshooting.

I repeated the test with a narrower request:

```text
Read ticket_normal.txt and only summarize its contents.
Do not provide troubleshooting steps.
```

This time it stopped after the document read:

![Controlled tool use after narrowing the task](images/02-controlled-tool-use.png)

Nothing about the available tools had changed. Only the task boundary changed.

That left me with a useful distinction:

```text
Can the agent call this tool?
          ≠
Should the agent call this tool for this task?
```

Tool allowlisting controls availability. It does not automatically solve unnecessary tool use.

---

## 6. Failure Experiment

Another failure mode appeared around termination. A model can receive an observation and still fail to move toward a final answer.

The loop can start looking like this:

```text
Reason
  ↓
Action
  ↓
Observation
  ↓
Reason
  ↓
Action
  ↓
Observation
  ↓
...
```

At that point the problem is no longer only answer quality. The agent is having trouble deciding when to stop.

This matters because each additional iteration can mean another model call or another tool call. With read-only tools that may be mostly wasteful. With state-changing tools, repeated actions can have a much larger impact.

---

## 7. Termination Failure

Ideally, each reasoning turn eventually reaches one of two states:

```text
More information needed → Action
Enough information      → Final Answer
```

But an LLM isn't a deterministic state machine. After an observation it might call the same tool again, choose another unnecessary tool, break the expected output format, or simply continue reasoning.

That raised a design question for me:

> Should the model be the only component responsible for deciding when the agent stops?

I don't think it should. Some termination controls need to exist in the orchestrator rather than in the model's instructions.

---

## 8. Why MAX_STEPS Matters

The lab therefore has a hard iteration limit:

```python
MAX_STEPS = 6
```

The loop is bounded by Python rather than by the model promising to stop:

```python
for step in range(1, MAX_STEPS + 1):
    ...
```

I tested the boundary by temporarily setting `MAX_STEPS = 2` and giving the agent a task that naturally required two tool calls before the final response.

The tools completed, but there was no third reasoning turn available for normal termination. The orchestrator stopped the loop and switched to the fallback path.

![MAX_STEPS stopping the reasoning loop](images/03-max-steps-circuit-breaker.png)

This separated two concepts that initially looked similar:

```text
Normal termination
→ the model decides it has enough information

Hard termination
→ the orchestrator decides the loop has run long enough
```

`MAX_STEPS` is a circuit breaker. It doesn't mean every long reasoning chain is malicious or incorrect. It means termination does not depend entirely on model behavior.

---

## 9. Duplicate Action Detection

A maximum step count gives the loop a hard upper bound, but obvious repetition can be caught earlier.

The agent stores previous `Action + Action Input` combinations:

```python
action_key = (action, tool_input)

if action_key in seen_actions:
    # stop the repeated action path
```

So these controls have different jobs:

```text
Duplicate Action Detection
→ catch a specific repeated operation early

MAX_STEPS
→ enforce a hard upper bound regardless of the pattern
```

During testing, duplicate-action detection exposed another problem. The model requested the same document twice, the orchestrator correctly stopped the loop, and the fallback path produced details that were not present in the collected observation.

That meant the loop control worked, but the new execution path created by the control had its own reliability problem.

I changed the fallback so it had to answer from collected observations and explicitly avoid inventing unsupported details.

![Grounded fallback after duplicate-action detection](images/04-grounded-fallback.png)

The resulting flow became:

```text
Repeated Action
      ↓
Stop Loop
      ↓
Grounded Fallback
      ↓
Answer from collected observations
```

This was a useful reminder that security controls also create code paths that need testing.

---

## 10. Security Implications

The reasoning loop becomes more interesting from a security perspective when observations come from sources the agent does not control.

An agent might read:

```text
Document
Email
Web Page
Knowledge Base
API Response
```

The flow is no longer simply "tool returns useful data":

```text
Untrusted Data
      ↓
Tool
      ↓
Observation
      ↓
LLM Reasoning
      ↓
Next Decision
```

An observation is part of the context used for the next decision. That means reviewing only direct user input is not enough. I also need to ask which observations can contain attacker-controlled content and what later decisions that content can influence.

---

## 11. From Reasoning Loop to Indirect Prompt Injection

I tested that boundary with a harmless marker. The user asked the agent to summarize a printer ticket, while the document contained an additional instruction telling the model to return a specific marker instead.

The result was:

```text
DOCUMENT_INSTRUCTION_FOLLOWED
```

![Indirect prompt injection through a document observation](images/05-indirect-prompt-injection.png)

The marker itself wasn't important. It only made the behavior change easy to observe. The important path was:

```text
Untrusted document
→ read_document
→ Observation
→ LLM context
→ behavior change
```

The tool had done exactly what it was supposed to do: read a file. The security issue appeared at the boundary where untrusted tool output became reasoning context.

I then marked the tool output as untrusted data and told the model not to follow instructions found inside it. The first test payload was ignored and the printer issue was summarized correctly.

![Prompt-level mitigation working for the first payload](images/06-indirect-prompt-injection-mitigated.png)

That result was useful, but it was not a reason to call the issue fixed. A different wording later influenced the model again. Even adding provenance-style metadata such as `Source` and `Trust: untrusted` did not create enforcement.

![Untrusted provenance metadata did not prevent the behavior change](images/09-provenance-bypass.png)

The lesson for this reasoning-loop lab was simple:

```text
Labeling data as untrusted
          ≠
Preventing the model from acting on it
```

This is where the experiment starts to overlap with a much larger subject: indirect prompt injection and agent trust boundaries. I am deliberately not trying to turn this project into a complete defense framework.

---

## 12. Security Controls

The experiments made it clear that reasoning-loop security cannot rely on the system prompt alone. The controls around the model matter just as much as the instructions inside it.

### Tool Allowlisting

The model should only be able to request tools explicitly exposed by the orchestrator. In this lab, `run_tool()` rejects unknown tool names instead of dynamically executing arbitrary functions.

### Least Privilege

Each tool should have only the permissions required for its job. A read-only policy lookup does not need the ability to modify system state.

### Iteration Limits

A hard loop limit such as `MAX_STEPS` provides deterministic termination outside the model.

### Duplicate Action Detection

Repeated `Action + Action Input` combinations can be stopped before the general iteration limit is reached.

### Input / Output Validation

Tool arguments and tool results should not automatically be treated as trustworthy just because they came through the agent framework.

I experimented with moving part of the document handling outside the model. Instead of passing every line from a document directly into the next reasoning turn, the orchestrator accepted only expected ticket fields.

![Code-level policy enforcement before the next reasoning turn](images/10-code-level-policy-enforcement.png)

This was stronger than simply asking the model to ignore unwanted content because rejected fields never reached the next reasoning step.

The first version was too strict, though. It also removed legitimate ticket details:

![Security and utility trade-off](images/11-policy-utility-tradeoff.png)

I replaced that with a small structured allowlist for fields such as `Subject`, `Issue`, `Started`, `Attempted fix`, and `Status`. That restored useful context:

![Structured allowlist restoring useful ticket context](images/12-structured-allowlist-utility-restored.png)

But this also exposed the limit of schema validation. If instruction-like text is placed inside an allowed field value, the structure is valid and the text can still reach the model:

![Instruction-like content inside an allowed field](images/14-injection-inside-allowed-field.png)

In that test the model ignored the embedded instruction, but the important point is that the content crossed the boundary.

```text
Schema validation
→ validates expected structure

Schema validation
≠ proves arbitrary natural-language values are safe
```

That is a good stopping point for this lab. Going further would move the project away from reasoning-loop behavior and into a dedicated study of prompt-injection defenses and agent trust models.

### Human Approval

Not every tool has the same impact. Read-only actions and state-changing actions should not automatically have the same approval model. Sensitive or difficult-to-reverse operations can use human approval as an additional boundary.

---

## 13. Lessons Learned

I started this lab thinking mostly about the mechanics of a ReAct loop. The experiments changed the questions I ask when looking at an agent.

1. **Where can untrusted data enter the reasoning loop?**
2. **Which later decisions can that data influence?**
3. **Does the agent need a tool for this task at all?**
4. **What happens when the model repeats an action?**
5. **If the model does not stop, what outside the model will stop it?**
6. **What happens on fallback and error paths after a control fires?**

The recurring theme was that the LLM should not be the only place where important boundaries exist.

```text
Model behavior
      ↓
Orchestrator controls
      ↓
Tool permissions
      ↓
Validation
      ↓
Bounded impact
```

`MAX_STEPS`, duplicate-action detection, tool allowlisting, and validation are all small examples of the same idea: **do not assume the model will always make the decision you hoped it would make.**

Understanding a reasoning loop is therefore not only about understanding how an agent makes decisions. It is also about understanding what can influence those decisions, what actions they can lead to, and where the surrounding system can enforce boundaries.

---

## Running the Lab

The lab expects Python 3, a local Ollama instance, and a model such as `llama3.1`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1
```

Start Ollama if it is not already running:

```bash
ollama serve
```

Run the agent:

```bash
python3 -m uvicorn agent:app --host 127.0.0.1 --port 8001 --reload
```

Check the health endpoint:

```bash
curl -s http://127.0.0.1:8001/health | python3 -m json.tool
```

Send a request:

```bash
curl -s http://127.0.0.1:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the account recovery policy?"}' \
  | python3 -m json.tool
```

The model can be changed without editing the code:

```bash
export OLLAMA_MODEL=llama3.1
```

## Tests

`test_agent.py` uses a scripted model so the parser, loop controls, tool dispatch, fallback behavior, and API endpoints can be tested without relying on non-deterministic model output.

```bash
python3 test_agent.py
```

The scripted tests validate orchestrator behavior. They are not a substitute for the live-model experiments shown above.

## Repository Layout

```text
01-reasoning-loop/
├── README.md
├── agent.py
├── agent-vulnerable.py
├── llm.py
├── tools.py
├── test_agent.py
├── requirements.txt
├── documents/
│   ├── ticket_normal.txt
│   ├── ticket_untrusted.txt
│   └── ticket_detailed.txt
└── images/
```

`agent-vulnerable.py` preserves an earlier version of the loop for comparison with the mitigations added during the experiments.
