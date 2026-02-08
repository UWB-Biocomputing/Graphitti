---
agent: "agent"
description: "Generate concise Google Test unit tests for Graphitti (C++17)"
---

## Task

Generate or update Google Test unit tests for the selected C++ function/method.

## Project context

- Graphitti is a C++17 neural simulation project.
- Tests live in `Testing/UnitTesting/` and use Google Test.

## Before you write tests

1. Check for existing tests:
   - `Testing/UnitTesting/{ClassName}Tests.cpp`
   - `Testing/UnitTesting/{ClassName}Test.cpp`
   - `Testing/UnitTesting/Test{ClassName}.cpp`
2. If tests exist, scan for errors and coverage gaps (missing methods, edge cases, error paths, branches).
3. Only add what is missing; do not duplicate tests.

## What to generate

- 5-8 focused tests covering:
  - Core behavior
  - Boundary/invalid inputs (nullptr, empty containers, min/max)
  - Error handling (exceptions or error returns)
  - Resource ownership or side effects if relevant
- Prefer behavior checks over implementation details.
- Use realistic Graphitti data where possible.

## Test structure

- Use `TEST()` or `TEST_F()` and AAA (Arrange, Act, Assert).
- Use `EXPECT_*` for non-fatal checks and `ASSERT_*` when continuation is unsafe.
- Use clear names: `TEST(ClassName, Method_Condition_Expected)`.

## Output

- Place tests in `Testing/UnitTesting/{ClassName}Tests.cpp`.
- If updating existing tests, mark new additions with a short comment.

## Inputs

Target function: ${input:function_name:Which C++ function or method should be tested?}
Target class: ${input:class_name:Which class does this function belong to? (or 'standalone' for free functions)}
