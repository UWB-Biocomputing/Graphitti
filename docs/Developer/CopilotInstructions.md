# GitHub Copilot Custom Instructions

## Overview

The `copilot-instructions.md` file is a special configuration file that allows the development team to tailor GitHub Copilot's behavior specifically for this repository. It functions as a **System Prompt**—context that is silently appended to every interaction you have with Copilot Chat or inline code generation within this project.

Instead of generic coding assistance, this file forces the AI to adopt our specific coding styles, architecture patterns, and contribution guidelines automatically.

## File Location

[.github/copilot-instructions.md](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/copilot-instructions.md)

> [!NOTE]
> This specific path is required by GitHub for the instructions to be automatically detected.

## How It Works

When you ask Copilot a question or ask it to generate code:

1. Copilot scans the repository context.
2. It reads `.github/copilot-instructions.md`.
3. It prioritizes rules defined in this file over its general training data.

For example, if the general training data suggests using `std::cout` for C++, but our instructions specify a custom logger class, Copilot will default to the custom logger.

## Structure & What to Include

This file is written in standard Markdown. To maintain effectiveness, it should be concise and focused on high-impact rules. Recommended sections include:

### 1. High-Level Context

Briefly explain what the software does (e.g., "A low-latency network simulator"). This helps the AI understand variable naming context and performance constraints.

### 2. Technology Stack

Explicitly list versions and tools.

- **Good:** "Use C++17 standards. Build system is CMake 3.20+."
- **Why:** This prevents the AI from suggesting C++20 features we cannot compile or C++98 legacy patterns we want to avoid.

### 3. Coding Style & Conventions

Define the "personality" of the code.

- **Naming:** CamelCase vs. snake_case.
- **Formatting:** Indentation rules, bracket placement.
- **Idioms:** "Always use smart pointers," "Avoid raw loops," etc.

### 4. Project-Specific Knowledge

List architectural details that an outsider (or AI) wouldn't know.

- Folder structure explanations.
- Key libraries (e.g., "Use strict types from the internal `Types` library, not primitives").
- Testing frameworks used.

## How to Edit and Maintain

As the project evolves, this file must be updated to prevent the AI from giving outdated advice.

- **When to update:**
  - When bumping compiler versions (e.g., C++17 to C++20).
  - When introducing a new major dependency.
  - When the team decides to change a styling convention.
- **Best Practices:**
  - **Be Explicit:** Do not be vague. Instead of "Write good code," say "Write code that passes `clang-tidy` checks."
  - **Keep it Updated:** If Copilot consistently makes the same mistake, add a rule here to correct it.

## External Resources

- [GitHub Docs: Configuring GitHub Copilot Custom Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)
- [GitHub Blog: How to use Copilot Custom Instructions](https://github.blog/changelog/2024-02-08-custom-instructions-for-github-copilot-in-vs-code/)
