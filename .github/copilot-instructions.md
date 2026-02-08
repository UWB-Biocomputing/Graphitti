# Copilot PR Review Instructions (Graphitti)

Start every review with a one-line acknowledgement that these Graphitti instructions are being used.

## Project context

- Graphitti is a high-performance C++17 simulator for graph-based systems (CPU and GPU).
- Build system: CMake. Tests: Google Test plus custom regression tests.
- Dependencies live under `ThirdParty/`.

## Code style essentials

- Use `.clang-format` from repo root.
- Indentation: 3 spaces, no tabs.
- Line length: 100 columns.
- Naming: camelCase; classes start uppercase, functions/vars lowercase.
- Braces: cuddled for control flow, isolated for function bodies; always use braces.
- Files: `.cpp`, `.h`, `.cu`; file names match primary class names (case-sensitive).
- Header guards: use `#pragma once`.

## C++ design essentials

- Target C++17; prefer `using` over `typedef`.
- Use `const`/`constexpr` (const first) and `const` methods where applicable.
- Explicitly declare or delete copy/move operations.
- Prefer smart pointers; avoid `auto_ptr`.
- Inputs: value or const ref; outputs: refs; optional inputs: `std::optional`.
- Use `override` on overrides; avoid `virtual` on overrides.
- Accessors return references for non-primitive members (const when read-only).

## PR review checklist (prioritize high-risk issues)

- Formatting and naming compliance with `.clang-format` and rules above.
- Correctness, safety (memory/threading), and error handling.
- Performance regressions in graph operations or memory usage.
- Tests updated/added for new behavior; existing tests still pass.
- Documentation updated for public API or user-facing changes.

## Branch and issue hygiene

- Target branch is `development` (not `master`).
- Branch name: `issue-####-short-description`.
- PR title: `[ISSUE-####] ...` and issue referenced in description.

## Review output

- Be concise and constructive.
- Provide concrete fixes or suggestions and cite specific lines.
