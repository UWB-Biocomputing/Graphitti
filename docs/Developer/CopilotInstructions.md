# GitHub Copilot Custom Instructions

## Overview

The `copilot-instructions.md` file is a configuration file that tailors GitHub Copilot's behavior for this repository. It acts as a **system prompt** — context that is automatically included in every Copilot Chat interaction, inline code generation, and **pull request code review** within this project.

Instead of receiving generic coding assistance, Copilot follows Graphitti's specific coding standards, architecture patterns, and conventions automatically — both when writing code and when reviewing it.

## File Location

[.github/copilot-instructions.md](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/copilot-instructions.md)

> [!NOTE]
> This path is required by GitHub for automatic detection. The file must be at `.github/copilot-instructions.md` in the repository root.

## Prerequisites

To use this file, you need:

1. **GitHub Copilot extension** installed in VS Code (see the [Copilot setup guide](https://code.visualstudio.com/docs/copilot/setup)).
2. **The correct workspace open** — you must open the repository root folder in VS Code (the folder that directly contains `.github/`). If `.github/` is not visible as a top-level folder in the Explorer sidebar, the instructions file will not be detected.
3. **VS Code 1.104 or later** — custom instructions are enabled by default in modern versions. On older versions, you may need to set `github.copilot.chat.codeGeneration.useInstructionFiles` to `true` in your VS Code settings.

## How It Works

When you interact with Copilot in this repository:

1. Copilot reads `.github/copilot-instructions.md` automatically.
2. The content is prepended as context to every chat message and code generation request.
3. Copilot prioritizes these rules over its general training data.

**Example:** Copilot's training data defaults to `std::cout` for C++ output. Because our instructions specify log4cplus, Copilot will use `LOG4CPLUS_INFO(...)` instead when generating code in this repository.

> [!IMPORTANT]
> The instructions file is **not** a prompt you invoke manually. It applies silently to every Copilot interaction. For task-specific prompts you run on demand (like generating unit tests), see the [Prompt Templates documentation](CopilotPromptTemplate.md).

## Pull Request Code Review

The same `copilot-instructions.md` file also guides **Copilot's pull request reviews** on GitHub.com. This means Copilot will enforce Graphitti's coding standards, performance constraints, and testing requirements when reviewing PRs — not just when generating code locally.

### How It Works

When Copilot reviews a pull request, it reads the `copilot-instructions.md` file from the repository's default branch and uses those rules to guide its analysis. For example, our instructions tell Copilot to flag unnecessary object copying and missing `CMakeLists.txt` updates, so PR reviews will surface these issues automatically.

### Requesting a Copilot Review

1. Create or open a pull request on GitHub.com.
2. Open the **Reviewers** menu on the right sidebar.
3. Select **Copilot** as a reviewer.
4. Wait for Copilot to analyze the changes (usually under 30 seconds).
5. Read through Copilot's inline comments on the PR diff.

> [!NOTE]
> Copilot always leaves a "Comment" review — it does not "Approve" or "Request Changes." This means Copilot reviews do not count toward required approvals and will not block merging.

### Automatic Reviews

Repository administrators can configure Copilot to automatically review all new pull requests without manual assignment. This is configured in the repository settings under **Copilot** → **Code review**. When enabled, every new PR will receive a Copilot review based on the instructions file.

### Why This Matters

Without custom instructions, Copilot's PR reviews are generic — it catches obvious issues but misses project-specific standards. With our instructions file, Copilot will:

- Flag performance issues specific to the simulator hot path (`Simulator/Core/`).
- Enforce Graphitti's 3-space indentation, CamelCase naming, and brace placement rules.
- Check that new logic includes corresponding unit tests in `Testing/UnitTesting/`.
- Verify `CMakeLists.txt` is updated when source files are added.
- Warn against modifications to `ThirdParty/` code.

This provides a consistent baseline review on every PR before human reviewers even look at the code.

### Enabling Custom Instructions for PR Reviews

Custom instructions for code review are enabled by default. If Copilot is not following the instructions during PR reviews:

1. Go to the repository **Settings** on GitHub.com.
2. Navigate to the **Copilot** section → **Code review**.
3. Ensure the option to use custom instructions is turned **on**.

## Instructions File vs. Prompt Files

These two file types serve different purposes:

|                   | `copilot-instructions.md`                                           | `.prompt.md` files                                                                    |
| ----------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Location**      | `.github/copilot-instructions.md`                                   | `.github/prompts/`                                                                    |
| **When applied**  | Automatically on every interaction (chat, code gen, and PR reviews) | Only when invoked via `/` slash command                                               |
| **Purpose**       | Project-wide coding standards and context                           | Task-specific workflows (e.g., generate tests, review code)                           |
| **Content style** | Short, imperative rules                                             | Detailed step-by-step instructions with personas and examples                         |
| **Example**       | "Use 3-space indentation"                                           | "Analyze the selected code, plan 5–7 test scenarios, then generate Google Test cases" |

## What to Include

The file should be concise and focused on high-impact rules. Effective instruction files contain five key sections:

### 1. Project Overview

A brief description of what the software does. This helps Copilot understand naming context, performance constraints, and domain-specific terminology.

### 2. Tech Stack

Explicitly list languages, versions, and tools. This prevents Copilot from suggesting incompatible features (e.g., C++20 features when the project uses C++17) or inventing non-existent build commands.

### 3. Code Standards

Define formatting and style rules that differ from common defaults. Focus on rules Copilot is likely to get wrong without guidance:

- Indentation size (especially non-standard values like 3 spaces)
- Naming conventions (CamelCase vs. snake_case)
- Brace placement
- Required attributes (`[[nodiscard]]`, `override`, `const`)

### 4. Architecture Map

List the directory structure and explain what each area is for. This allows Copilot to suggest correct file paths for new code and understand which areas have strict performance requirements.

### 5. Review and Testing Priorities

Define what Copilot should check during code reviews and what testing standards apply to new code.

## How to Edit and Maintain

As the project evolves, this file must be updated to prevent Copilot from giving outdated advice.

**When to update:**

- When bumping compiler or language versions (e.g., C++17 to C++20).
- When introducing a new major dependency or removing one.
- When the team changes a styling convention.
- When Copilot consistently makes the same mistake — add a rule to correct it.

**Best practices:**

- **Be explicit and imperative.** Write "Use 3-space indentation" instead of "We prefer 3-space indentation." Write "Do not use `printf`" instead of "Avoid `printf` when possible."
- **Include the "why" for non-obvious rules.** For example: "Always use braces on single-line blocks to prevent dangling-else bugs during maintenance." This helps Copilot make correct decisions in ambiguous situations.
- **Keep it concise.** The entire file is sent as context on every interaction. Long files consume tokens that could otherwise be used for your actual question or code. Aim for under 100 lines.
- **Move specialized rules to `.instructions.md` files.** If a rule only applies to certain file types (e.g., CUDA-specific rules for `.cu` files), put it in a separate file under `.github/instructions/` with an `applyTo` glob pattern rather than in the global instructions file.
- **Test with real PRs.** After updating the instructions, open a pull request and request a Copilot review to verify the new rules are being applied. Iterate based on the results.

## External Resources

- [VS Code: Custom Instructions Documentation](https://code.visualstudio.com/docs/copilot/customization/custom-instructions)
- [GitHub Docs: Adding Repository Custom Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)
- [GitHub Docs: Using Copilot Code Review](https://docs.github.com/en/copilot/using-github-copilot/code-review/using-copilot-code-review)
- [GitHub Blog: 5 Tips for Writing Better Custom Instructions](https://github.blog/ai-and-ml/github-copilot/5-tips-for-writing-better-custom-instructions-for-copilot/)
- [GitHub Blog: Unlocking the Full Power of Copilot Code Review](https://github.blog/ai-and-ml/unlocking-the-full-power-of-copilot-code-review-master-your-instructions-files/)
- [GitHub Tutorial: Custom Instructions for Code Review](https://docs.github.com/en/copilot/tutorials/use-custom-instructions)
