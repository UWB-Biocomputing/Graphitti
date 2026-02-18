# Copilot Prompt Templates: Unit Test Generation

## Overview

The file located at `.github/prompts/generate-unit-tests.prompt.md` serves as a **Prompt Template**. Unlike the global instructions file, this is a specialized "recipe" used to execute a specific task—in this case, generating robust unit tests.

This technique is often referred to as "Prompt Engineering." It provides the AI with a strict persona, a specific workflow, and examples (few-shot prompting) to ensure that generated tests match the project's quality standards.

## File Location

[.github/prompts/generate-unit-tests.prompt.md](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/prompts/generate-unit-tests.prompt.md)

## Why Use a Prompt Template?

Asking Copilot to simply "write a test for this function" often results in:

- Generic or brittle tests.
- Inconsistent naming conventions.
- Testing implementation details rather than behavior.

By invoking this template, we force Copilot to:

1.  **Analyze** the target code first.
2.  **Plan** the test cases (success paths, edge cases, error handling).
3.  **Generate** code that matches our specific testing framework (e.g., Google Test) and directory structure.

## Anatomy of the Prompt File

To edit or create new prompt templates, follow this structure:

### 1. The Persona

Define who the AI is.

> _Example: "You are a Senior QA Engineer specializing in C++ Google Test..."_

### 2. The Context & Constraints

Define the rules of the road.

- **Naming:** How should test files and test cases be named?
- **Placement:** Where do the files go? (e.g., `Testing/UnitTesting/`).
- **Style:** AAA (Arrange, Act, Assert) pattern, usage of `ASSERT` vs `EXPECT`.

### 3. The Workflow

Step-by-step instructions for the AI to follow internally.

1.  Search for existing tests (to avoid duplicates).
2.  Identify boundary conditions.
3.  Draft the code.

### 4. Few-Shot Examples (Crucial)

The most effective way to guide the AI is to show, not just tell. Include snippet examples of:

- **Input:** A target function signature.
- **Output:** The perfect unit test code for that function, formatted exactly how we want it.

## Usage Guide

To utilize this prompt in your workflow:

1.  Open Copilot Chat in your IDE.
2.  Reference the file (e.g., `@workspace`) or copy the prompt contents.
3.  Provide the **Input Variables**:
    - **Target Class/Function:** What you want to test.
4.  Copilot will generate the output based on the template's logic.

## Maintenance

Update this file when:

- **Framework Changes:** We switch testing libraries (e.g., Google Test to Catch2).
- **Process Changes:** We require new sections in tests (e.g., requiring performance benchmarks in unit tests).
- **Quality Issues:** If Copilot frequently misses edge cases, add an explicit step in the "Workflow" section to "Analyze integer overflows" or "Check for null pointers."

## External Resources

- [GitHub Copilot: Prompt Engineering for Developers](https://docs.github.com/en/copilot/using-github-copilot/prompt-engineering-for-github-copilot)
- [Microsoft: Introduction to Prompt Engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering)
