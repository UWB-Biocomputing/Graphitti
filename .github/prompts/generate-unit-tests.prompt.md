---
agent: "agent"
description: "Generate Google Test unit tests for C++ functions or methods in Graphitti"
---

## What is a Unit Test?

A **unit test** is a type of software testing that verifies the correctness of individual, isolated components (or "units") of code—typically functions, methods, or classes.

### Key Characteristics

- **Isolated**: Tests a single piece of functionality in isolation from the rest of the system
- **Fast**: Should execute quickly (milliseconds)
- **Repeatable**: Produces the same result every time
- **Independent**: Doesn't depend on other tests or external systems (databases, networks, files)

### Purpose

1. **Catch bugs early** – Find issues before they propagate
2. **Enable refactoring** – Safely change code knowing tests will catch regressions
3. **Document behavior** – Tests serve as executable documentation of how code should work

---

## What is Google Test (gtest)?

**Google Test** is a popular open-source C++ testing framework developed by Google for writing and running unit tests.

### Key Features

- **Rich assertions** – Provides `EXPECT_*` (non-fatal) and `ASSERT_*` (fatal) macros
- **Test discovery** – Automatically finds and runs tests
- **Test fixtures** – Share setup/teardown code between tests using `TEST_F()`
- **Death tests** – Verify code crashes or exits as expected
- **Parameterized tests** – Run the same test with different inputs
- **XML/JSON output** – Integrates with CI/CD systems

### EXPECT vs ASSERT

| Type       | Behavior on Failure                                    |
| ---------- | ------------------------------------------------------ |
| `EXPECT_*` | Records failure but continues test execution           |
| `ASSERT_*` | Records failure and stops the current test immediately |

Use `EXPECT_*` when you want to see all failures in a test. Use `ASSERT_*` when continuing doesn't make sense (e.g., a null pointer would crash subsequent code).

---

## Task

Analyze the selected C++ function/method and generate or improve Google Test unit tests that thoroughly validate its behavior.

## Project Context

This is the Graphitti project - a neural network simulator. Tests are located in `Testing/UnitTesting/` and use the Google Test framework.

---

## Pre-Generation Check: Existing Test Analysis

**Before generating new tests, perform these steps:**

### Step 1: Check for Existing Test File

Look for an existing test file at:

- `Testing/UnitTesting/{ClassName}Tests.cpp`
- `Testing/UnitTesting/{ClassName}Test.cpp`
- `Testing/UnitTesting/Test{ClassName}.cpp`

### Step 2: If Tests Exist, Analyze Them

If a test file already exists, **analyze it before generating new tests**:

#### 2a. Error Analysis

- Check for syntax errors or incorrect Google Test macro usage
- Identify deprecated assertions or patterns
- Look for tests that may produce false positives/negatives
- Verify proper use of `EXPECT_*` vs `ASSERT_*`
- Check for memory leaks or improper resource handling in tests
- Identify flaky test patterns (timing-dependent, order-dependent)

#### 2b. Coverage Gap Analysis

Compare the existing tests against the source code to identify:

- **Untested public methods** – Methods with no corresponding test cases
- **Missing edge cases** – Boundary conditions not covered
- **Missing error paths** – Exception handling or error returns not tested
- **Missing input variations** – Only happy path tested, no invalid inputs
- **Untested branches** – Conditional logic not fully exercised

#### 2c. Report Findings

Provide a summary:

```
## Existing Test Analysis: {ClassName}Tests.cpp

### Errors Found
- [List any errors or issues]

### Coverage Gaps
| Method/Function | Status | Missing Coverage |
|-----------------|--------|------------------|
| methodName()    | Partial | Missing nullptr test, boundary cases |
| otherMethod()   | None    | No tests exist |

### Recommendations
1. [Specific improvements needed]
```

### Step 3: Generate or Update Tests

Based on the analysis:

- **No existing tests**: Generate complete test suite
- **Tests exist with errors**: Fix errors and add missing coverage
- **Tests exist with gaps**: Generate only the missing test cases, clearly marked as additions

---

## Test Generation Strategy

1. **Core Functionality Tests**
   - Test the main purpose/expected behavior
   - Verify return values with typical inputs
   - Test with realistic simulation data scenarios

2. **Input Validation Tests**
   - Test with nullptr/null pointers
   - Test with empty containers (vectors, maps)
   - Test boundary values (min/max array indices, zero, negative numbers)
   - Test invalid enum values or out-of-range parameters

3. **Error Handling Tests**
   - Test expected exceptions are thrown (use `EXPECT_THROW`, `ASSERT_THROW`)
   - Verify error states are handled gracefully
   - Test edge cases specific to simulation parameters

4. **Resource Management Tests** (if applicable)
   - Verify proper memory allocation/deallocation
   - Test RAII patterns work correctly
   - Validate GPU/device memory handling for CUDA code

5. **Side Effects Tests** (if applicable)
   - Verify external calls are made correctly
   - Test state changes in simulation objects
   - Validate interactions with factory classes and managers

## Test Structure Requirements

- Use Google Test framework (`gtest/gtest.h`)
- Use `TEST()` for standalone tests, `TEST_F()` for fixture-based tests
- Follow AAA pattern: Arrange, Act, Assert
- Use descriptive test names: `TEST(ClassName, MethodName_Condition_ExpectedResult)`
- Group related tests in test fixtures when sharing setup/teardown
- Use `EXPECT_*` for non-fatal assertions, `ASSERT_*` for fatal assertions

## Google Test Assertion Reference

```cpp
// Equality
EXPECT_EQ(expected, actual);
EXPECT_NE(val1, val2);

// Boolean
EXPECT_TRUE(condition);
EXPECT_FALSE(condition);

// Comparisons
EXPECT_LT(val1, val2);   // less than
EXPECT_LE(val1, val2);   // less than or equal
EXPECT_GT(val1, val2);   // greater than
EXPECT_GE(val1, val2);   // greater than or equal

// Floating point (handles precision issues)
EXPECT_FLOAT_EQ(expected, actual);
EXPECT_DOUBLE_EQ(expected, actual);
EXPECT_NEAR(val1, val2, abs_error);

// Strings
EXPECT_STREQ(expected, actual);   // C-strings equal
EXPECT_STRNE(str1, str2);         // C-strings not equal

// Exceptions
EXPECT_THROW(statement, exception_type);
EXPECT_NO_THROW(statement);
EXPECT_ANY_THROW(statement);

// Pointers
EXPECT_EQ(nullptr, ptr);
EXPECT_NE(nullptr, ptr);
```

## Test File Template

```cpp
// {ClassName}Tests.cpp

#include "gtest/gtest.h"
#include "{HeaderFile}.h"
// Include other necessary headers

// Test fixture for tests requiring shared setup
class {ClassName}Test : public ::testing::Test {
protected:
   void SetUp() override {
      // Initialize test objects before each test
   }

   void TearDown() override {
      // Clean up resources after each test
   }

   // Shared test data members
};

// Standalone test example
TEST({ClassName}Test, MethodName_ValidInput_ReturnsExpected) {
   // Arrange - set up test data

   // Act - call the function under test

   // Assert - verify the results
}

// Fixture-based test example
TEST_F({ClassName}Test, MethodName_EdgeCase_HandlesCorrectly) {
   // Arrange - fixture provides shared setup

   // Act

   // Assert
}
```

## Input Parameters

Target function: ${input:function_name:Which C++ function or method should be tested?}
Target class: ${input:class_name:Which class does this function belong to? (or 'standalone' for free functions)}

## Guidelines

- **Always check for existing tests first** before generating new ones
- Generate 5-8 focused test cases covering the most important scenarios
- When updating existing tests, clearly mark new additions with comments
- Include realistic test data matching Graphitti simulation parameters
- Add comments explaining complex test setup or non-obvious assertions
- Ensure tests are independent and can run in any order
- Focus on testing behavior, not implementation details
- Consider GPU/CUDA implications if testing device code
- Reference existing test patterns in the UnitTesting folder
- Place output files in `Testing/UnitTesting/{ClassName}Tests.cpp`

Create tests that give confidence the function works correctly and help catch regressions in the neural simulation.
