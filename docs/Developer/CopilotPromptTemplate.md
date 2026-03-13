# Copilot Prompt Template Anatomy

## Table of Contents

- [Overview](#overview)
- [Why Use a Prompt Template?](#why-use-a-prompt-template)
- [Standard File Structure](#standard-file-structure)
- [YAML Frontmatter](#yaml-frontmatter)
  - [Common Fields](#common-fields)
- [Workflow Layout](#workflow-layout)
- [Few-Shot Guidance](#few-shot-guidance)
- [Input Variables](#input-variables)
- [Maintenance](#maintenance)
- [Related Documentation](#related-documentation)

## Overview

This document describes the shared structure used for Graphitti Copilot prompt files.

Use this as the common reference for prompt template design. Prompt-specific behavior and examples
should remain in each prompt's own documentation file.

## Why Use a Prompt Template?

Asking Copilot “why is this failing?” or "Generate some test cases for this." with no structure often yields vague guesses, incomplete analysis, or suggestions that ignore how your project is actually wired together.

By using a template with a concrete structure, you force Copilot to:

1. **Analyze** the reported problem and the relevant code before answering, using a set of internal questions about expected vs. actual behavior, invariants, and likely bug categories.
2. **Trace** the execution path from the entry point using repository search and file reads.
3. **Explain** what steps it will take to solve your problem.

This leads to more reproducible, reviewable outputs and makes it easier for other contributors to understand and validate the diagnosis.

## Standard File Structure

Each `.prompt.md` file should include the following sections:

1. **YAML frontmatter** for prompt metadata and execution mode.
2. **Workflow instructions** that define the step-by-step reasoning process.
3. **Few-shot example** that demonstrates expected output style and quality.
4. **Input variables** that describe how context is passed into the prompt.

## YAML Frontmatter

Frontmatter is enclosed in `---` blocks at the top of the prompt file. Here is an example frontmatter:

```yaml
---
name: your-command-name
description: Short summary shown in slash-command autocomplete
agent: agent
tools: ["search", "read", "edit"]
---
```

### Common Fields

| Field         | Purpose                                                                                                               |
| ------------- | --------------------------------------------------------------------------------------------------------------------- |
| `name`        | Slash command name used in chat (for example, `/generate-unit-tests`).                                                |
| `description` | Short summary shown in prompt-file and slash-command UI.                                                              |
| `agent`       | Execution mode (`agent`, `ask`, or `plan`) based on expected behavior.                                                |
| `tools`       | Allowed tools (for example, `search`, `read`, `edit`) used by the prompt run. These are only utilized in `agent` mode |
| `model`       | Optional model override. Omit unless a prompt requires a specific model.                                              |

A full list of YAML fields can be found [here](https://code.visualstudio.com/docs/copilot/customization/prompt-files#_prompt-file-format).

A full list of usable chat tools can be found [here](https://code.visualstudio.com/docs/copilot/reference/copilot-vscode-features#_chat-tools), or by opening the **Chat view**, selecting **Agent** from the agent picker, and then clicking the **Configure Tools** button in the chat input field.

## Workflow Layout

Prompt workflows should use a clear, sequential structure such as:

1. **Understand** the target code/problem and identify dependencies and constraints.
2. **Plan** scenarios or steps before generating output.
3. **Generate** final output that follows project standards.

This pattern keeps prompt behavior predictable and improves output quality across prompt types.

## Few-Shot Guidance

The most effective way to guide the AI is to show, not just tell. Prompt files should include concrete examples, for example:

- **Input:** A class `Counter` with methods `increment()`, `decrement()`, and `getCount()`.
- **Output:** Three complete Google Test cases (`IncrementIncreasesCount`, `DecrementFromZeroDoesNotGoNegative`, `IncrementThenDecrementReturnsToOriginal`) demonstrating the expected formatting, the AAA pattern, fixture usage, and inline comments.

This example anchors the style for all generated tests.

Remember, few-shot examples should stay short and representative rather than exhaustive.

## Input Variables

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

A full list of all possible input variables can be found [here](https://code.visualstudio.com/docs/reference/variables-reference).

## Maintenance

Update prompt files when:

- **Project structure changes:** File paths, module names, or directory layouts change in ways that affect how you want functions and tests referenced.
- **Processes changes:** You want additional sections in a report (for example, a “Performance Impact” or “Regression Risk” section), or different internal analysis questions.
- **Quality issues appear:** If Copilot repeatedly misses certain bug patterns (such as concurrency issues or floating-point precision problems), refine the internal questions or add explicit guidance in the workflow.
- **Tooling changes:** If you switch Copilot models or want to let the debug prompt create patches automatically, you may change `agent` from `ask` to `agent` and add `edit` to the `tools` list.

When you make changes, keep the structure predictable:

- Preserve the YAML frontmatter format.
- Keep the three main steps (Understand, Plan, Generate), even if you add more detail inside them.
- Maintain at least one up-to-date few-shot example that reflects your current conventions.

## Related Documentation

- For installation and environment setup, see [GitHub Copilot Setup in VS Code](CopilotSetup.md).
- For unit-test prompt specifics, see [Copilot Prompt Templates: Unit Test Generation](CopilotGenerateUnitTests.md).
- For debug prompt specifics, see [Copilot Prompt Templates: Debugging and Root Cause Analysis](CopilotDebug.md).
