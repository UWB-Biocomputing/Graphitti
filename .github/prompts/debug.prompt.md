---
name: debug-code
description: Trace and explain the root cause of a bug in the C++17 Graphitti code
agent: agent
tools: ["search", "read"]
---

# Inputs and Context

Use the following inputs as the primary source of truth for expected vs. actual behavior. Treat them as ground truth unless you explicitly state otherwise in your assumptions.

- **Problem description from the user (required):**

  ${input:problemDescription}

- **Target code (optional selection from the editor):**

  ${selection}

- **Steps to reproduce (optional):**

  ${input:stepsToReproduce}

- **Logs, stack traces, or failing test output (optional):**

  ${input:logs}

- **Environment / configuration details (optional):**

  ${input:environment}

If any of these are missing or incomplete, call that out explicitly in the **Assumptions and Scope** section instead of silently guessing.

# Step 1: Understand the Code and Problem

Before proposing any fix, read and summarize the target code and the reported behavior. Answer these questions internally (do not output them):

1. **What is the primary function, method, or code path under suspicion?** Is this a class (`Graph`, `Vertex`), a free function, or a template?
2. **What is the expected behavior versus the actual behavior?** Include inputs, outputs, and any side effects.
3. **What dependencies does it have?** Other Graphitti classes, standard library containers, external libraries?
4. **What invariants or assumptions are relevant?** (e.g., "vertex count must equal the size of the adjacency list", "counter should increase by exactly 1").
5. **What categories of bug are most likely?** (logic error, off-by-one, stale or shared state, wrong constant, incorrect default, misuse of a dependency, etc.).

Use the **Inputs and Context** section above as your starting point. Do **not** output your answers to these questions directly. Use them only to guide the trace and explanation in later steps.

# Step 2: Trace the Execution Path

Now that you understand the problem, trace how the code executes from the entry point of the bug to the final incorrect result.

Use the `search` and `read` tools to locate and inspect any relevant files, definitions, and call sites in the Graphitti codebase (for example, where a method is defined and where it is called).

Design an internal execution trace that answers:

1. Which public or entry-point function is called when this bug occurs?
2. Which functions, methods, or constructors are invoked along the way (including helpers, virtual overrides, and templates)?
3. How do the key values change at each step (inputs, member variables, return values, accumulators, caches)?
4. Which branches or conditions are taken in the failing scenario?
5. Where is the first point at which the state diverges from what is expected?

Do not output this raw trace verbatim. In the next step, you will convert it into a clear explanation for the user.

When exploring the repository:

- Only reference functions, classes, and files that you have actually located with `search` / `read`.
- If you cannot find a symbol, file, or configuration that seems important, note this as a limitation in **Assumptions and Scope** rather than inventing its behavior.

# Step 3: Explain the Root Cause and Fix

Using the analysis from Step 1 and the trace from Step 2, generate a structured debugging report following these rules:

## Project Conventions

- Organize the answer into the following sections, in this exact order:
  1. **Problem Summary** — Restate the bug in 1–3 sentences, including the relevant function(s) and the expected vs. actual behavior.
  2. **Assumptions and Scope** — Briefly list any assumptions you are making about inputs, configuration, or environment, including any missing information from the Inputs and Context section.
  3. **Execution Trace** — Summarize the key steps in the call chain that lead to the incorrect result, focusing on functions, important state changes, and branches taken.
  4. **Root Cause Analysis** — Explain precisely _why_ the bug happens, referencing specific functions, conditions, or state transitions.
  5. **Proposed Fix** — Describe a minimal, targeted change that would correct the behavior, including a concrete code snippet or patch-style suggestion when appropriate.
  6. **Verification Steps** — Outline how to verify the fix (existing tests to run, new tests to add, and any manual steps).

- In **Execution Trace**, summarize only the essential steps, state changes, and branches relevant to the failing scenario. Avoid line-by-line commentary or speculative paths that are not actually taken when the bug occurs.
- Only reference functions, classes, files, and configuration values that you have actually seen in the repository or in the provided inputs. If something is inferred or assumed, make that explicit in **Assumptions and Scope**.
- If there are multiple plausible root causes or the available information is incomplete, clearly:
  - Identify the most likely root cause first.
  - List alternative possibilities as secondary hypotheses, along with what extra data would be needed to confirm or reject them.
- Use concise paragraphs and bullet points. Avoid long, unstructured walls of text.
- When referencing functions or methods, include both the name and file path where possible (e.g., ``Counter::increment` in `Simulator/Utils/Counter.cpp``).

## File Placement

- You are **not** required to create or edit files directly in this prompt. Focus on analysis and explanation.
- When you propose code changes, show them as **snippets** the user can apply manually, rather than editing files yourself.
- If you reference tests or new test files, follow the existing project structure:
  - Unit tests live under `Testing/UnitTesting/`.
  - For functions touched by the fix, you may suggest using the `/generate-unit-tests` prompt on the affected code to create new tests that reproduce the bug and guard against regressions.

## Code Style

- Keep code examples minimal and focused on the bug and its fix.
- Use C++17 features and the project’s conventions when writing example code (e.g., `auto` where appropriate, no `using namespace std;`).
- When showing a fix, include only the smallest necessary surrounding context (e.g., the function signature plus the corrected lines).
- Add brief inline comments explaining _why_ the fix works, especially when adjusting constants, conditions, or state transitions.

## Few-Shot Example

Below is an example of the expected output format. Match this style exactly, adapting it to the actual bug and code under analysis.

**Input:** Problem description — “`Counter::increment()` always returns 2 when it should return 1 after a single call.” The user selects the implementation of `Counter::increment()` and its helper `internalAdd`.

**Output:**

````md
1. Problem Summary

`Counter::increment()` increases the internal counter by 2 instead of 1 on a single call.  
Expected: calling `increment()` once from an initial count of 0 should yield 1.  
Actual: the count becomes 2.

2. Assumptions and Scope

- We assume `Counter` starts at 0 unless explicitly initialized otherwise.
- We focus on the non-thread-safe, single-threaded usage of `Counter`.
- No external configuration or environment flags modify the increment step.

3. Execution Trace

- Step 1: `Counter::increment()` is called while `count_` is 0 (`Simulator/Utils/Counter.cpp`, ~line 25).
- Step 2: Inside `increment()`, it calls `internalAdd(2)` instead of `internalAdd(1)`.
- Step 3: `internalAdd(2)` updates `count_` from 0 to 2 and returns.
- Step 4: `getCount()` simply returns `count_` without further modification, so the observed result is 2.

4. Root Cause Analysis

The root cause is a hard-coded increment value of 2 inside `Counter::increment()`.  
The intended behavior is to increase the count by exactly 1 per call, but the call to `internalAdd(2)` doubles the expected increment.  
There are no additional side effects or state mutations that compensate for this, so the incorrect value propagates directly to `getCount()`.

5. Proposed Fix

Change the argument passed to `internalAdd` from 2 to 1:

```cpp
void Counter::increment() {
    // Increment by 1 to match the documented behavior.
    internalAdd(1);
}
```

This ensures a single call to increment() increases count\_ by exactly 1.

6. Verification Steps

Add or update a unit test that:

- Constructs a Counter at 0.
- Calls increment() once.
- Asserts that getCount() returns 1.

Run the existing unit test suite (for example, ctest or your standard test command) to confirm no unrelated regressions.

Optionally, use the /generate-unit-tests prompt on Counter to generate additional tests that:

- Call increment() twice and confirm the result is 2.
- Interleave increment() and decrement() and ensure the count returns to the original value.
````

Now generate the debugging report for the target problem and selection.
