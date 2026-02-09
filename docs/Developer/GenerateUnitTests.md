# Generate Unit Tests Prompt

## Overview

The `generate-unit-tests.prompt.md` file located in `.github/prompts/generate-unit-tests.prompt.md` is a GitHub Copilot prompt template that guides AI agents in generating Google Test unit tests for Graphitti C++17 code. This prompt ensures consistent, high-quality test generation that follows Graphitti's testing conventions and best practices.

## Purpose

This prompt template helps developers:

- Generate comprehensive unit tests for Graphitti classes and functions
- Maintain consistent test naming and structure across the codebase
- Follow Google Test best practices and Graphitti conventions
- Avoid duplicate test generation by checking existing tests first
- Create tests that cover edge cases, boundary conditions, and error handling

## File Location

```
.github/prompts/generate-unit-tests.prompt.md
```

## How to Use

### Invoking the Prompt

In VS Code with GitHub Copilot:

1. Open the file containing the function or class you want to test
2. Invoke the Copilot command palette
3. Select "Generate Unit Tests" or type `@workspace /generate-unit-tests`
4. Provide the requested inputs:
   - **Target function**: The name of the function or method to test
   - **Target class**: The class name (or 'standalone' for free functions)

### What Happens

The prompt guides Copilot through a multi-step process:

1. **Discovery**: Searches for existing tests to avoid duplication and identify coverage gaps
2. **Research**: Launches a subagent to locate the target code, dependencies, and existing test patterns
3. **Test Plan**: Proposes 5-8 focused test cases covering:
   - Core functionality
   - Boundary conditions
   - Invalid inputs
   - Error handling
   - Resource management
4. **Implementation**: Generates tests following Graphitti conventions

## Test Generation Guidelines

### Naming Conventions

- **Classes**: `TEST(ClassName, FunctionalityBeingTested)`
- **Free Functions**: `TEST(ModuleName, FunctionalityBeingTested)`
- Use PascalCase throughout; no underscores

Examples:

```cpp
TEST(Matrix, MultiplicationProducesCorrectResult)
TEST(AllVertices, CreatesDefaultConstructedVertex)
TEST(UtilityFunctions, ParsesConfigurationCorrectly)
```

### File Placement

- **Classes**: `Testing/UnitTesting/{ClassName}Tests.cpp`
- **Free Functions**: `Testing/UnitTesting/{ModuleName}Tests.cpp` or `Testing/UnitTesting/UtilityTests.cpp`

### Test Structure

All tests follow the AAA (Arrange, Act, Assert) pattern:

```cpp
TEST(ClassName, FunctionalityBeingTested) {
   // Arrange: Set up test data and preconditions
   MyClass instance;
   int expectedValue = 42;

   // Act: Execute the functionality being tested
   int result = instance.computeValue();

   // Assert: Verify the results
   EXPECT_EQ(result, expectedValue);
}
```

### Assertion Guidelines

- Use `EXPECT_*` for non-fatal checks (test continues after failure)
- Use `ASSERT_*` when continuation is unsafe (test stops immediately on failure)
- Test behavior, not implementation details

### Coverage Goals

Generated tests should cover:

- **Core behavior**: Normal, expected usage with typical inputs
- **Boundary conditions**: Empty containers, min/max values, edge cases
- **Invalid inputs**: Null pointers, out-of-range values, malformed data
- **Error handling**: Exceptions, error codes, failure modes
- **Resource management**: Memory ownership, lifetime, cleanup

## Example Workflow

### For a Class Method

**Input:**

- Function: `advance`
- Class: `Simulator`

**Generated Tests (examples):**

```cpp
TEST(Simulator, AdvanceIncreasesCurrentStep) { ... }
TEST(Simulator, AdvanceWithZeroStepsDoesNothing) { ... }
TEST(Simulator, AdvanceWithNegativeStepsThrowsException) { ... }
TEST(Simulator, AdvanceUpdatesAllVertices) { ... }
TEST(Simulator, AdvanceWithLargeStepCountMaintainsStability) { ... }
```

### For a Free Function

**Input:**

- Function: `parseConfigFile`
- Class: `standalone`

**Generated Tests (examples):**

```cpp
TEST(ConfigParser, ParsesValidConfigurationFile) { ... }
TEST(ConfigParser, RejectsEmptyFilePath) { ... }
TEST(ConfigParser, ThrowsExceptionOnMalformedXML) { ... }
TEST(ConfigParser, HandlesOptionalParametersCorrectly) { ... }
```

## Best Practices

### Integration with Existing Tests

- Always check for existing test files before generating new ones
- Add new tests to existing files when appropriate
- Include a brief comment marking new additions (e.g., `// Added: Test for boundary condition`)
- Follow the organization and style of surrounding tests

### Realistic Test Data

- Use data patterns consistent with Graphitti simulations
- Reference actual configuration files or data structures when possible
- Avoid magic numbers; use named constants or explain values in comments

### Incremental Development

- Generate tests for one function or method at a time
- Run tests after generation to verify they compile and pass
- Iterate on coverage gaps identified during testing

## Maintenance

When Graphitti's testing conventions or Google Test usage patterns evolve, update the prompt template to reflect these changes. This ensures newly generated tests remain consistent with the codebase.

## Related Documentation

- [Unit Testing Documentation](UnitTests.md)
- [Google Tests Tutorial](GoogleTestsTutorial.md)
- [C++ Style Guide](cppStyleGuide.md)
- [Coding Conventions](codingConventions.md)

---

[<< Go back to Developer Documentation](index.md)
