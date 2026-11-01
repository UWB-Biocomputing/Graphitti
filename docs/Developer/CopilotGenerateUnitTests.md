# Copilot Prompt Templates: Unit Test Generation

## Table of Contents

- [Overview](#overview)
- [File Location](#file-location)
- [Prerequisites: Setting Up Copilot in VS Code](#prerequisites-setting-up-copilot-in-vs-code)
- [Prompt Template Format](#prompt-template-format)
- [Unit Test Prompt Format](#unit-test-prompt-format)
  - [Step 1: Understand the Code](#step-1-understand-the-code)
  - [Step 2: Create Scenario](#step-2-create-scenario)
  - [Step 3: Generate Test Cases](#step-3-generate-test-cases)
- [What Happens Next: How Copilot Responds](#what-happens-next-how-copilot-responds)
- [Example Workflow](#example-workflow)
- [External Resources](#external-resources)

## Overview

The file located at `.github/prompts/generate-unit-tests.prompt.md` serves as a **Prompt Template**. Unlike the global instructions file, this is a specialized "recipe" used to execute a specific task—in this case, generating robust unit tests.

This technique is often referred to as "Prompt Engineering." It provides the AI with a structured workflow and examples (few-shot prompting) to ensure that generated tests match the project's quality standards.

## File Location

This prompt file is located under the `.github/prompts/` directory in the repository root [here](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/prompts/generate-unit-tests.prompt.md).

## Prerequisites: Setting Up Copilot in VS Code

View the setup instructions in the [CopilotSetup.md](./CopilotSetup.md) file.

## Prompt Template Format

See why we use templates and how to structure them in the [CopilotPromptTemplate.md](./CopilotPromptTemplate.md) file

## Unit Test Prompt Format

This is how the `generate-unit-tests.prompt.md` file specifically is structured:

### Step 1: Understand the Code

Copilot first reads the selected class/function and identifies test-relevant behavior:

1. Public API surface to test (methods, inputs, return values).
2. Preconditions and invariants that should hold before and after operations.
3. Observable behaviors versus implementation details to avoid brittle tests.
4. Error paths and boundary conditions (empty collections, min/max values, invalid inputs).
5. Dependencies or collaborators that may require fixtures or controlled setup.

### Step 2: Create Scenario

Next, Copilot creates a concrete scenario set for unit testing rather than root-cause tracing:

- Happy path behavior.
- Boundary and edge-case coverage.
- Error handling and failure expectations.
- State preservation across operations.
- Repeated-call/idempotency behavior.
- Method interaction sequences (for example, add/remove or increment/decrement flows).

### Step 3: Generate Test Cases

Finally, Copilot generates test code that follows Graphitti testing conventions:

1. Uses Google Test with clear `TEST`/`TEST_F` names in PascalCase.
2. Prefers behavior-focused assertions (`EXPECT_*`/`ASSERT_*`) and AAA-style structure.
3. Places or appends tests under `Testing/UnitTesting/`.
4. Uses fixtures only where setup complexity justifies them.
5. Produces tests that are isolated and deterministic.

## What Happens Next: How Copilot Responds

Because this prompt is configured with `agent: agent`, Copilot proposes direct workspace edits instead of
only chat text.

You should expect:

1. Creation or modification of unit-test files (typically under `Testing/UnitTesting/`).
2. A diff you can review and accept/discard in VS Code.
3. Follow-up refinement by prompting for additional scenarios (for example, edge cases or regressions).

## Example Workflow

**Scenario:** Generate tests for the `Vertex` class.

1. Open `Simulator/Core/Vertex.cpp` in the editor.
2. Select the class methods you want tested (or press `Ctrl+A` to select all).
3. Open Copilot Chat and type `/generate-unit-tests`

4. Copilot reads the selected code via the `${selection}` variable, follows the prompt's three-step workflow (analyze → plan → generate), and creates a new file at `Testing/UnitTesting/VertexTests.cpp` containing Google Test cases.
5. Review the generated diff. Click **Accept** to save, or type follow-up instructions in the chat to refine.

## External Resources

- [VS Code: Prompt Files Documentation](https://code.visualstudio.com/docs/copilot/customization/prompt-files)
- [GitHub Copilot: Prompt Engineering for Developers](https://docs.github.com/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Microsoft: Introduction to Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
- [Awesome Copilot: Community Prompt Examples](https://github.com/github/awesome-copilot)
