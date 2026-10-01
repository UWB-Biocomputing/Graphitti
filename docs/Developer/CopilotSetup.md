# GitHub Copilot Setup in VS Code

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [1. Install and Sign In](#1-install-and-sign-in)
  - [2. Open the Correct Workspace Folder](#2-open-the-correct-workspace-folder)
  - [3. Verify Prompt File Detection](#3-verify-prompt-file-detection)
- [Using Prompt Files](#using-prompt-files)
- [Understanding the `@` Syntax and `#` Context](#understanding-the--syntax-and--context)
- [Related Documentation](#related-documentation)

## Overview

This document describes how to install and configure GitHub Copilot for use in Graphitti.

## Instructions

### 1. Install and Sign In

1. Open VS Code.
2. Go to the **Extensions** sidebar (`Ctrl+Shift+X` on Windows/Linux, `Cmd+Shift+X` on Mac).
3. Search for **"GitHub Copilot Chat"** and install it.
4. Ensure you are signed in with a GitHub account that has an active Copilot subscription (free tier, Pro, or through an organization).
   - **How to check your login status:** Look at the bottom-right corner of the VS Code Status Bar for the **Copilot icon**.  If the icon is visible and doesn't have a warning badge or a slash through it, you are actively logged in. You can also click the **Accounts** (profile gear) icon in the bottom-left corner to verify your active GitHub session.
   - **How to log in manually:** If you weren't prompted to log in automatically upon installation, simply click the **Accounts** icon (bottom-left) or the **Copilot** icon (bottom-right) and select **Sign in to use Copilot**.

### 2. Open the Correct Workspace Folder

Prompt files are resolved **relative to the workspace root** — the folder you open in VS Code.

> **Important:** You must open the repository root folder that directly contains the `.github/prompts/` directory. For Graphitti, this means opening the folder that has `.github/` as a direct child.
>
> For example, if the repository is cloned to `/home/user/Graphitti/`, open the **Graphitti** folder in VS Code — not **/home or /user**. If VS Code's Explorer sidebar shows `.github/` as a top-level folder, you're in the right place.

### 3. Verify Prompt File Detection

1. Open the Copilot Chat panel (`Ctrl+Alt+I` on Windows/Linux, `Cmd+Ctrl+I` on Mac).
2. Click the **Configure Chat** gear icon (⚙) at the top of the Chat panel.
3. Select **Prompt Files** from the menu.
4. You should see available prompt files (for example, `generate-unit-tests` and `debug`) listed. If they appear, setup is complete.

If it does not appear, confirm:

- The file is named with the `.prompt.md` extension (not just `.md`).
- The file is inside `.github/prompts/` at the workspace root.
- You are on VS Code version **1.104 or later** (prompt files are enabled by default in modern versions).

## Using Prompt Files

You may run prompt files through multiple VS Code entry points listed below:

| Method                  | How                                                                                                                        |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| **Slash command** | Type `/command-name` in the Chat input.                                                                                    |
| **Command Palette** | `Ctrl+Shift+P` (Win/Linux) or `Cmd+Shift+P` (Mac) → **Chat: Run Prompt** → select `command-name`.                          |
| **Play button** | Open `command-name.prompt.md` in the editor and click the ▶ play button in the editor title bar.                           |
| **Configure Chat menu** | Click ⚙ in the Chat view → **Prompt Files** → select the `command-name` prompt.                                            |

## Understanding the `@` Syntax and `#` Context

You may see references to `@workspace` or `#codebase` in Copilot documentation. Here is what they mean:

- **`@workspace`** — A built-in _chat participant_ that gives Copilot knowledge of your entire project. When you type `@workspace` followed by a question, Copilot searches across all files in your workspace to answer it. Example: `@workspace Where is the Graph class defined?`
- **`@terminal`** — A chat participant for terminal-related questions. Example: `@terminal How do I run the tests?`
- **`@vscode`** — A chat participant for VS Code settings and features. Example: `@vscode How do I change the font size?`
- **`#codebase`** — A _context tool_ that adds codebase search results to your prompt. Unlike `@workspace` (which handles the entire prompt), `#codebase` can be combined with other tools. It is the recommended approach for adding project-wide context.
- **`#file`** — Attaches a specific file as context. Example: `#file:Vertex.h Explain this class.`

You generally do **not** need to use `@workspace` or `#codebase` when running prompt files, because those prompts are already configured with tools that give Copilot automatic access to your codebase (the exact tools vary by prompt).

## Related Documentation

- For shared prompt file anatomy and workflow layout, see [Copilot Prompt Template Anatomy](CopilotPromptTemplate.md).
- For unit-test prompt details, see [Copilot Prompt Templates: Unit Test Generation](CopilotGenerateUnitTests.md).
- For debug prompt details, see [Copilot Prompt Templates: Debugging and Root Cause Analysis](CopilotDebug.md).
