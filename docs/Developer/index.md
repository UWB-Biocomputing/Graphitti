# Developer Documentation

If you're developing Graphitti code, then here are your reference documents.

Writing new code? Then make sure to follow our [contributing guide] and _document your code here_.

Reading code that isn't obvious? When you figure out how it works, then _document it here_ and _document it in comments in the code._

## Student Quick Start

Students, use this [quickstart guide](StudentSetup.md) to help setup, use, and develop with Graphitti.

## Software Development Process

- To further understand our development process, please check out our [gitflow documentation](GitFlow.md).

- Your pull requests will not be approved if you do not adhere to our [coding conventions](codingConventions.md).

## Graphitti Repository Tools and Workflows

- CMake
  - Refer to the [CMake](CMake.md) documentation to help with any related CMake questions
- clang-format
  - Refer to the [clang-format documentation](codingConventions.md#clang-format) to help with using this tool
- GitHub Copilot
  - [Copilot Prompt Template](CopilotPromptTemplate.md) - Why we use .prompt.md files and how they are structured.
  - [Copilot Setup](CopilotSetup.md) - How to configure GitHub Copilot for use with VS Code and Graphitti
  - [Copilot Instructions](CopilotInstructions.md) - How the copilot-instructions.md file is configured for Graphitti development
  - [Generate Unit Tests](CopilotGenerateUnitTests.md) - Using the AI-assisted unit test generation prompt
  - [Debug](CopilotDebug.md) - Using Copilot to assist with debugging
- GitHub Pages
  - Refer to the [GitHub Pages documentation](GHPages.md) section for an overview of how we use GitHub Pages and editing practices
- GitHub Actions Workflows
  - We have a [Doxygen and GitHub Pages Action](GHActions.md#doxygen-and-github-pages-action-gh-pagesyml) to regenerate and publish documentation automatically or manually
  - We have a [Code Style Action](GHActions.md#code-style-check-formatyml) to verify formatting with clang-format
  - We have a [Unit Tests Action](GHActions.md#unit-tests-unit-testsyml) to run unit tests
  - We have a [Regression Tests Action](GHActions.md#regression-tests-regression-testsyml) to run simulation regression tests
  - We have an [Auto-Close Merged Issues Action](GHActions.md#auto-close-merged-issues-close-merged-issuesyml) to automatically close issues when pull requests merge into `SharedDevelopment` or `master`
- Repository Maintenance Scripts
  - [Stale Issue Cleanup Script](GHActions.md#stale-issue-cleanup-cleanup_stale_issuessh) to audit and batch-close issues resolved in merged pull requests

## Graphitti System Documentation

- Diagrams
  - Here is a list of [UML class diagrams](classDiagrams.md) (in Mermaid) of Graphitti
  - Here are the [sequence diagrams](sequenceDiagrams.md) (in Mermaid) for the Graphitti system
- Doxygen
  - Documentation generated from source code
  - Doxygen provides web-based indices and hierarchical views of Graphitti's class and file structures
  - [Visit Doxygen Generated Documentation]
  - Document code in the `.h` file using the [Doxygen Style Guide](../Doxygen/DoxygenStyleGuide.md) format
  - [Doxygen Update Guide](../Doxygen/DoxygenUpdateGuide.md)
- [Event buffering](eventBuffering.md) in vertex classes.
- [Performing Analyses](PerformingAnalyses.md)
- [Neuro Implementation](NeuroImplementation.md)
- [GraphManager and InputManager classes](GraphAndEventInputs.md)
- [Configuration](../User/configuration.md)

---

[<< Go back to the Graphitti home page](../index.md)

[//]: # "Moving URL links to the bottom of the document for ease of updating - LS"
[//]: # "Links to repo items which exist outside of the docs folder need an absolute link."
[contributing guide]: https://github.com/UWB-Biocomputing/Graphitti/blob/master/CONTRIBUTING.md
[Visit Doxygen Generated Documentation]: https://uwb-biocomputing.github.io/Graphitti/Doxygen/html/index.html
