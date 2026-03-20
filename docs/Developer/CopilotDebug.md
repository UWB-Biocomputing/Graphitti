# Copilot Prompt Templates: Debugging Graphitti Bugs

## Table of Contents

- [Overview](#overview)
- [File Location](#file-location)
- [Prerequisites: Setting Up Copilot in VS Code](#prerequisites-setting-up-copilot-in-vs-code)
- [Prompt Template Format](#prompt-template-format)
- [Debug Prompt Format](#debug-prompt-format)
  - [Step 1: Understand the Code and Problem](#step-1-understand-the-code-and-problem)
  - [Step 2: Trace the Execution Path](#step-2-trace-the-execution-path)
  - [Step 3: Explain the Root Cause and Fix](#step-3-explain-the-root-cause-and-fix)
- [What Happens Next: How Copilot Responds](#what-happens-next-how-copilot-responds)
- [Example Workflow](#example-workflow)
- [External Resources](#external-resources)

## Overview

The file `debug.prompt.md` serves as a **Prompt Template** for GitHub Copilot. Unlike global instructions, this is a specialized “recipe” used to perform a focused task—tracing and explaining the root cause of bugs in the C++17 Graphitti codebase.

This prompt guides Copilot through a structured debugging workflow: understanding the reported problem, tracing the execution path through the code, and producing a clear, sectioned explanation of the root cause and a proposed fix.

## File Location

This prompt file is located under the `.github/prompts/` directory in the repository root [here](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/prompts/debug.prompt.md).

## Prerequisites: Setting Up Copilot in VS Code

View the setup instructions in the [CopilotSetup.md](./CopilotSetup.md) file.

## Prompt Template Format

See why we use templates and how to structure them in the [CopilotPromptTemplate.md](./CopilotPromptTemplate.md) file

## Debug Prompt Format

This is how the `debug.prompt.md` file specifically is structured:

### Step 1: Understand the Code and Problem

Copilot first reads the reported problem and any selected code, then answers several **internal** questions (not shown to the user) about:

1. The primary function, method, or code path under suspicion (class, free function, template, etc.).
2. Expected vs. actual behavior, including inputs, outputs, and side effects.
3. Dependencies (other Graphitti classes, standard library containers, external libraries).
4. Relevant invariants or assumptions (for example, “vertex count must equal the size of the adjacency list”).
5. Likely bug categories (logic error, off-by-one, stale state, wrong constant, etc.).

### Step 2: Trace the Execution Path

Next, Copilot uses `search` and `read` to explore the repository and build an internal execution trace from the entry point to the incorrect result:

- Which public or entry-point function is called when the bug occurs.
- Which functions, methods, or constructors are invoked along the way, including helpers and overrides.
- How key values change at each step (inputs, member variables, return values, accumulators, caches).
- Which branches or conditions are taken in the failing scenario.
- The first point where the state diverges from what is expected.

### Step 3: Explain the Root Cause and Fix

Finally, Copilot produces a structured report with six sections in a fixed order:

1. **Problem Summary** — Concise restatement of the bug, expected vs. actual behavior, and relevant functions.
2. **Assumptions and Scope** — Any assumptions about inputs, configuration, or environment, and what information is missing.
3. **Execution Trace** — A summarized, human-readable version of the call chain and key state changes leading to the bug.
4. **Root Cause Analysis** — A precise explanation of why the bug occurs, tied to specific functions, conditions, or state transitions.
5. **Proposed Fix** — A minimal, targeted change with a code snippet or patch-style suggestion where appropriate.
6. **Verification Steps** — How to verify the fix (tests to run or add, plus any manual steps).

## What Happens Next: How Copilot Responds

Because this prompt is configured with `agent: agent`, but does not have the `edit` tool available, Copilot responds **only in the chat panel**; it does **not** create or edit files directly.

You should expect:

1. A structured markdown report with the six sections described above.
2. References to specific functions and (where possible) file paths and approximate line numbers.
3. One or more proposed fixes in code snippets, plus guidance on tests or manual checks to verify the fix.

You can then:

- Copy any suggested code into your source files manually.
- Run or add tests following the Verification Steps.
- Ask follow-up questions (e.g., “What if this function is also used by X?” or “Show me an alternative fix that doesn’t change this invariant.”).

## Example Workflow

**Scenario:** You see a bug where a spike count in a neural simulation is off by one for large graphs.

1. Open the file containing the suspected logic (for example, a neuron or synapse update function).
2. Select the relevant function or region (the spike accumulation code).
3. Open Copilot Chat and type:

```text
/debug-code The spike count is off by 1 when running large graphs with more than 10,000 vertices.
```

4. Copilot analyzes the selected code and repository using the debug prompt workflow, then returns a report with an execution trace and a proposed fix.
5. Apply or adapt the suggested fix, then run the tests listed in the Verification Steps to confirm the issue is resolved.

## External Resources

- [VS Code: Prompt Files Documentation](https://code.visualstudio.com/docs/copilot/customization/prompt-files)
- [GitHub Copilot: Prompt Engineering for Developers](https://docs.github.com/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Microsoft: Introduction to Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
- [Awesome Copilot: Community Prompt Examples](https://github.com/github/awesome-copilot)
