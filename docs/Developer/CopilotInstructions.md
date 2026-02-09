# Copilot Instructions

## Overview

The `copilot-instructions.md` file located in `.github/copilot-instructions.md` provides GitHub Copilot with essential context about the Graphitti project. This file serves as an onboarding guide for AI coding agents, ensuring they understand Graphitti's architecture, conventions, and development practices before assisting with code reviews, pull requests, or code generation.

## Purpose

This instruction file helps Copilot:

- Understand Graphitti's high-level architecture and purpose
- Follow the project's C++17 coding standards and style conventions
- Navigate the repository structure efficiently
- Apply appropriate testing and CI/CD practices
- Adhere to contribution guidelines and workflow requirements

## File Location

```
.github/copilot-instructions.md
```

## Key Sections

### Repository Summary

Provides a high-level understanding of Graphitti as a C++17 graph-based simulator for neuroscience and emergency communications modeling, including build system and testing framework information.

### Tech Stack and Validated Tool Versions

Lists the specific tools validated for Graphitti development:

- g++ (C++17 compiler)
- CMake (build system)
- clang-format (code formatting)
- Optional dependencies: CUDA, HDF5, Boost Graph library

### Project Layout

Maps the high-signal paths in the repository:

- `Simulator/`: Core simulator implementation
- `Testing/`: Unit and regression tests
- `ThirdParty/`: Vendored dependencies
- `Tools/`: Python utilities
- `docs/`: Documentation
- `build/`: CMake build output

### Style and C++ Standards

Enforces strict coding conventions:

- 3-space indentation, 100-column limit
- camelCase naming (classes uppercase, functions/vars lowercase)
- `#pragma once` header guards
- Modern C++17 practices (smart pointers, `constexpr`, etc.)

### CI and Validation

Documents the GitHub Actions workflows for:

- Unit and regression testing
- Code formatting validation
- Documentation generation
- Diagram updates

### Contribution Hygiene

Defines workflow requirements:

- Branch naming: `issue-####-short-description`
- PR title format: `[ISSUE-####] ...`
- No direct commits to `master`

## Usage

Copilot automatically reads this file when working in the Graphitti repository. Developers do not need to manually reference it during normal development work. The file ensures that Copilot-generated code and suggestions align with Graphitti's established practices.

## Maintenance

When updating project conventions, tools, or workflows, update the copilot-instructions.md file to keep Copilot's understanding current. This ensures consistent AI assistance across the project lifecycle.

## Verification

At the start of code reviews or pull request reviews, Copilot will indicate that it has been onboarded using this file, confirming that it has loaded and understood the project context.

---

[<< Go back to Developer Documentation](index.md)
