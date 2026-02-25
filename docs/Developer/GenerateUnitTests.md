# Copilot Prompt Templates: Unit Test Generation

## Overview

The file located at `.github/prompts/generate-unit-tests.prompt.md` serves as a **Prompt Template**. Unlike the global instructions file, this is a specialized "recipe" used to execute a specific task—in this case, generating robust unit tests.

This technique is often referred to as "Prompt Engineering." It provides the AI with a structured workflow and examples (few-shot prompting) to ensure that generated tests match the project's quality standards.

## File Location

[.github/prompts/generate-unit-tests.prompt.md](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/prompts/generate-unit-tests.prompt.md)

## Prerequisites: Setting Up Copilot in VS Code

Before using this prompt template, you must have GitHub Copilot installed and configured in VS Code.

### 1. Install the Extension

1. Open VS Code.
2. Go to the **Extensions** sidebar (`Ctrl+Shift+X`).
3. Search for **"GitHub Copilot"** and install it (this also installs the Copilot Chat component).
4. Sign in with your GitHub account when prompted. You need an active GitHub Copilot subscription (free tier, Pro, or through an organization).

### 2. Open the Correct Workspace Folder

Prompt files are resolved **relative to the workspace root** — the folder you open in VS Code.

> **Important:** You must open the repository root folder that directly contains the `.github/prompts/` directory. For Graphitti, this means opening the folder that has `.github/` as a direct child.
>
> For example, if the repository is cloned to `/home/user/Graphitti/`, open the **Graphitti** folder in VS Code — not **/home or /user**. If VS Code's Explorer sidebar shows `.github/` as a top-level folder, you're in the right place.

### 3. Verify Prompt File Detection

1. Open the Copilot Chat panel (`Ctrl+Alt+I`).
2. Click the **Configure Chat** gear icon (⚙) at the top of the Chat panel.
3. Select **Prompt Files** from the menu.
4. You should see `generate-unit-tests` listed. If it appears, setup is complete.

If it does not appear, confirm:

- The file is named with the `.prompt.md` extension (not just `.md`).
- The file is inside `.github/prompts/` at the workspace root.
- You are on VS Code version **1.104 or later** (prompt files are enabled by default in modern versions).

## Why Use a Prompt Template?

Asking Copilot to simply "write a test for this function" often results in:

- Generic or brittle tests.
- Inconsistent naming conventions.
- Testing implementation details rather than behavior.

By invoking this template, we force Copilot to:

1.  **Analyze** the target code first (identify the SUT, public methods, dependencies, invariants, and failure modes).
2.  **Plan** the test cases across six categories (happy path, boundary values, error handling, state preservation, idempotency, and method interactions).
3.  **Generate** code that matches our specific testing framework (Google Test) and directory structure.

## Anatomy of the Prompt File

To edit or create new prompt templates, follow this structure:

### 1. YAML Frontmatter

The top of every `.prompt.md` file has a YAML metadata block enclosed in `---`. This controls how the prompt behaves:

```yaml
---
name: generate-unit-tests
description: Generate comprehensive Google Test cases for C++17 Graphitti code
agent: agent
tools: ["search", "read", "edit"]
---
```

| Field         | Purpose                                                                                                                                              |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `name`        | The slash command name. This is what you type after `/` in chat (e.g., `/generate-unit-tests`). If omitted, the filename is used.                    |
| `description` | A short summary shown next to the command in the autocomplete menu.                                                                                  |
| `agent`       | Which Copilot mode runs the prompt. Use `agent` for tasks that create/edit files, `ask` for Q&A, or `plan` for generating a step-by-step plan.       |
| `tools`       | The tools Copilot is allowed to use. `search` finds files in the codebase, `read` reads file contents, and `edit` creates or modifies files.         |
| `model`       | _(Optional)_ A specific AI model ID (e.g., `gpt-4o`, `copilot-claude-sonnet-4`). Omit this to use whichever model is currently selected in the chat. |

### 2. The Workflow (Step-by-Step Instructions)

The body of the prompt file is a three-step workflow that Copilot follows internally when generating tests. Unlike a simple list of rules, this workflow forces the AI to think before it writes.

#### Step 1: Understand the Code

Before writing any tests, Copilot reads and summarizes the target code by answering five analysis questions internally (these are not output to the user):

1. **What is the SUT (System Under Test)?** — Is this a class, free function, or template?
2. **What are the public methods and their signatures?** — Parameters, return types, preconditions.
3. **What dependencies does it have?** — Other Graphitti classes, standard library containers, external libraries.
4. **What invariants does the class maintain?** — e.g., "vertex count must equal the size of the adjacency list."
5. **What can go wrong?** — Null pointers, empty containers, out-of-range indices, integer overflow, floating point precision.

The selected code is injected via the `${selection}` variable (see [Input Variables](#4-input-variables) below).

#### Step 2: Design the Test Plan

Copilot designs 5–7 test scenarios covering these six categories. For each scenario, it writes one sentence describing the test and the expected outcome:

1. **Happy Path** — The standard use case works as expected.
2. **Boundary Values** — Min/max values (0 nodes, max edges, empty containers, single-element collections).
3. **Error Handling** — Verify correct exceptions are thrown or error codes are returned for invalid input.
4. **State Preservation** — After an operation, the object is in a valid and expected state.
5. **Idempotency / Repeated Calls** — Calling a method twice produces consistent results.
6. **Interaction Between Methods** — A sequence of operations (e.g., add then remove) leaves the object in the correct state.

#### Step 3: Generate the Test Code

Using the analysis from Step 1 and the plan from Step 2, Copilot generates C++ test code following the project rules defined in three sub-sections:

- **Project Conventions** — PascalCase test names, no `using namespace std;`, `EXPECT_*` vs `ASSERT_*` guidance, and the AAA (Arrange, Act, Assert) pattern.
- **File Placement** — All tests go into `Testing/UnitTesting/`. Copilot appends to existing test files or creates new ones as needed.
- **Code Style** — Relative include paths, test fixtures for complex setup, C++17 features, and inline comments explaining _why_ each scenario is tested.

### 3. Few-Shot Example

The most effective way to guide the AI is to show, not just tell. The prompt file includes a concrete example:

- **Input:** A class `Counter` with methods `increment()`, `decrement()`, and `getCount()`.
- **Output:** Three complete Google Test cases (`IncrementIncreasesCount`, `DecrementFromZeroDoesNotGoNegative`, `IncrementThenDecrementReturnsToOriginal`) demonstrating the expected formatting, the AAA pattern, fixture usage, and inline comments.

This example anchors the style for all generated tests.

### 4. Input Variables

Input variables are placeholders in the prompt body that get replaced with real values when the prompt runs. They use the `${...}` syntax.

| Variable                            | What It Resolves To                                                                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `${selection}` or `${selectedText}` | The code you currently have highlighted in the editor.                                                                                                 |
| `${file}`                           | The full path to the currently open file.                                                                                                              |
| `${fileBasename}`                   | The filename only (e.g., `Vertex.cpp`).                                                                                                                |
| `${fileDirname}`                    | The directory containing the current file.                                                                                                             |
| `${fileBasenameNoExtension}`        | The filename without its extension (e.g., `Vertex`).                                                                                                   |
| `${workspaceFolder}`                | The root path of the open workspace.                                                                                                                   |
| `${input:variableName}`             | Prompts you to type a value when the command runs. For example, `${input:framework:jest or vitest}` shows an input box with the hint "jest or vitest". |

Our prompt file uses `${selection}` — this means Copilot will use whatever code you have selected in the editor as the target for test generation.

## Usage Guide

### Invoking the Prompt (Slash Command)

1. **Open a source file** you want to generate tests for (e.g., `Simulator/Core/Vertex.cpp`).
2. **Select the code** you want to test — highlight a function, a class, or an entire file's contents in the editor.
3. **Open Copilot Chat** (`Ctrl+Alt+I`).
4. **Type `/generate-unit-tests`** in the chat input. The prompt should appear in the autocomplete dropdown as you type. Press Enter to invoke it.
5. _(Optional)_ You can add extra instructions after the slash command, e.g.:
   ```
   /generate-unit-tests Focus on edge cases for empty graphs
   ```

### What Happens Next: How Copilot Generates Output

When using **Agent mode** (which this prompt file is configured for via `agent: agent`), Copilot does not just print text in the sidebar — it **directly creates and edits files in your workspace**. Here is what to expect:

1. **Copilot analyzes** your selected code and the prompt instructions.
2. **Copilot creates or edits files** — for example, it may create `Testing/UnitTesting/VertexTests.cpp` or append new test cases to an existing test file. You will see a diff view showing the proposed changes.
3. **You review the changes** — VS Code highlights every addition and modification. You can:
   - **Accept** the changes to keep them.
   - **Discard** individual changes or all changes.
   - **Iterate** by sending a follow-up message (e.g., "Add a test for null input").

> **Note:** If Copilot is set to `ask` mode instead of `agent` mode, it will only display the generated code as text in the chat sidebar. You would then need to manually copy the code into your files. The `agent` setting in our prompt file's frontmatter ensures Copilot uses agent mode, which can create/edit files directly.

### Full Example Walkthrough

**Scenario:** Generate tests for the `Vertex` class.

1. Open `Simulator/Core/Vertex.cpp` in the editor.
2. Select the class methods you want tested (or press `Ctrl+A` to select all).
3. Open Copilot Chat and type:
   ```
   /generate-unit-tests
   ```
4. Copilot reads the selected code via the `${selection}` variable, follows the prompt's three-step workflow (analyze → plan → generate), and creates a new file at `Testing/UnitTesting/VertexTests.cpp` containing Google Test cases.
5. Review the generated diff. Click **Accept** to save, or type follow-up instructions in the chat to refine.

### Alternative Ways to Run the Prompt

| Method                  | How                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------- |
| **Slash command**       | Type `/generate-unit-tests` in the Chat input.                                                          |
| **Command Palette**     | `Ctrl+Shift+P` → `Chat: Run Prompt` → select `generate-unit-tests`.                                     |
| **Play button**         | Open `generate-unit-tests.prompt.md` in the editor and click the ▶ play button in the editor title bar. |
| **Configure Chat menu** | Click ⚙ in the Chat view → **Prompt Files** → select the prompt.                                        |

## Understanding the `@` Syntax and `#` Context

You may see references to `@workspace` or `#codebase` in Copilot documentation. Here is what they mean:

- **`@workspace`** — A built-in _chat participant_ that gives Copilot knowledge of your entire project. When you type `@workspace` followed by a question, Copilot searches across all files in your workspace to answer it. Example: `@workspace Where is the Graph class defined?`
- **`@terminal`** — A chat participant for terminal-related questions. Example: `@terminal How do I run the tests?`
- **`@vscode`** — A chat participant for VS Code settings and features. Example: `@vscode How do I change the font size?`
- **`#codebase`** — A _context tool_ that adds codebase search results to your prompt. Unlike `@workspace` (which handles the entire prompt), `#codebase` can be combined with other tools. It is the recommended approach for adding project-wide context.
- **`#file`** — Attaches a specific file as context. Example: `#file:Vertex.h Explain this class.`

> **Note:** You do not need to use `@workspace` or `#codebase` when running our prompt file. The prompt is already configured with `tools: ['search', 'read', 'edit']`, which gives Copilot the ability to search and read your codebase automatically.

## Maintenance

Update this file when:

- **Framework Changes:** We switch testing libraries (e.g., Google Test to Catch2).
- **Process Changes:** We require new sections in tests (e.g., requiring performance benchmarks in unit tests).
- **Quality Issues:** If Copilot frequently misses edge cases, add an explicit step in the workflow or a new test category in Step 2.

## External Resources

- [VS Code: Prompt Files Documentation](https://code.visualstudio.com/docs/copilot/customization/prompt-files)
- [GitHub Copilot: Prompt Engineering for Developers](https://docs.github.com/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Microsoft: Introduction to Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
- [Awesome Copilot: Community Prompt Examples](https://github.com/github/awesome-copilot)
