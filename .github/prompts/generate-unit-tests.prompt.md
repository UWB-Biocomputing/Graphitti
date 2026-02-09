---
name: generate-unit-tests
description: Generate Google Test unit tests for Graphitti C++17 functions
argument-hint: Optionally specify the function, class, or module focus for tests
agent: agent
---

Generate Google Test unit tests for Graphitti C++17 code.

Target function: ${input:function_name:Which C++ function or method should be tested?}
Target class: ${input:class_name:Which class does this function belong to? (or 'standalone' for free functions)}

## Discovery

Search for existing tests and nearby usage before writing anything. Prioritize these locations:

- `Testing/UnitTesting/{ClassName}Tests.cpp`
- `Testing/UnitTesting/{ClassName}Test.cpp`
- `Testing/UnitTesting/Test{ClassName}.cpp`
- Adjacent module tests in `Testing/UnitTesting/` that cover similar behaviors

If tests exist, scan for coverage gaps and only add missing tests.

## Research

Start a subagent to locate the target class/function definition, its dependencies, and any test fixtures or helpers already in use. Capture: required includes, namespace usage, and typical test data patterns.

## Test Plan

Propose 5-8 focused test cases that cover:

- Core behavior and expected outputs
- Boundary conditions and invalid inputs
- Null/empty containers and min/max values
- Error handling or exceptions (when applicable)
- Resource ownership or lifetime behavior (when relevant)

## Output Guidelines

- Use Google Test (`TEST` or `TEST_F`) and the AAA pattern (Arrange, Act, Assert)
- Use `EXPECT_*` for non-fatal checks; `ASSERT_*` when continuation is unsafe
- Test behavior, not implementation details
- Follow naming: `TEST(ClassName, FunctionalityBeingTested)` (use PascalCase; no underscores)
- Place new tests in `Testing/UnitTesting/{ClassName}Tests.cpp`
- Include realistic Graphitti simulation data where appropriate
