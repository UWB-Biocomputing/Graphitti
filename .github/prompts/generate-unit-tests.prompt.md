---
name: generate-unit-tests
description: Generate comprehensive Google Test cases for C++17 Graphitti code
agent: agent
tools: ["search", "read", "edit"]
---

# Step 1: Understand the Code

Before writing any tests, read and summarize the target code. Answer these questions internally (do not output them):

1.  **What is the SUT (System Under Test)?** Is this a class (`Graph`, `Vertex`), a free function, or a template?
2.  **What are the public methods and their signatures?** List each method, its parameters, return type, and any preconditions.
3.  **What dependencies does it have?** Other Graphitti classes, standard library containers, external libraries?
4.  **What invariants does the class maintain?** (e.g., "vertex count must equal the size of the adjacency list")
5.  **What can go wrong?** Null pointers, empty containers, out-of-range indices, integer overflow, floating point precision.

Target Code:
${selection}

# Step 2: Design the Test Plan

Now that you understand the code, design 5–7 test scenarios covering these categories. For each scenario, write one sentence describing the test and the expected outcome.

1.  **Happy Path** — The standard use case works as expected.
2.  **Boundary Values** — Min/max values (0 nodes, max edges, empty containers, single-element collections).
3.  **Error Handling** — Verify correct exceptions are thrown or error codes are returned for invalid input.
4.  **State Preservation** — After an operation, the object is in a valid and expected state.
5.  **Idempotency / Repeated Calls** — Calling a method twice produces consistent results.
6.  **Interaction Between Methods** — A sequence of operations (e.g., add then remove) leaves the object in the correct state.

# Step 3: Generate the Test Code

Using the analysis from Step 1 and the plan from Step 2, generate the C++ test code following these rules:

## Project Conventions

- Use `PascalCase` for test names: `TEST(Graph, AddsVertexCorrectly)`.
- Do NOT use `using namespace std;`.
- Use `EXPECT_*` for assertions that should not abort the test. Use `ASSERT_*` for pointer validity or preconditions where continuing would crash.
- Use the AAA pattern: **Arrange** (setup), **Act** (call the method), **Assert** (verify the result). Separate each section with a blank line.

## File Placement

- All tests go into `Testing/UnitTesting/`.
- If a test file already exists (e.g., `Testing/UnitTesting/VertexTests.cpp`), generate **only the new test cases** to append.
- If no test file exists, generate the **entire file** including headers and fixture setup.

## Code Style

- Include necessary headers with relative paths (e.g., `#include "Simulator/Core/Vertex.h"`).
- If the class requires complex setup, create a test fixture: `class VertexTest : public ::testing::Test { ... }`.
- Use C++17 features where appropriate: `auto`, structured bindings, `std::optional`, `constexpr`.
- Add a brief inline comment on each test explaining **why** that specific value or scenario is being tested.

## Few-Shot Example

Below is an example of the expected output format. Match this style exactly.

**Input:** A class `Counter` with methods `increment()`, `decrement()`, and `getCount()`.

**Output:**

```cpp
#include <gtest/gtest.h>
#include "Simulator/Utils/Counter.h"

class CounterTest : public ::testing::Test {
protected:
   Counter counter_;

   void SetUp() override {
      counter_ = Counter();
   }
};

// Happy path: incrementing increases the count by 1
TEST_F(CounterTest, IncrementIncreasesCount) {
   counter_.increment();

   EXPECT_EQ(counter_.getCount(), 1);
}

// Boundary: decrementing from zero should not produce a negative count
TEST_F(CounterTest, DecrementFromZeroDoesNotGoNegative) {
   counter_.decrement();

   EXPECT_GE(counter_.getCount(), 0);
}

// State preservation: increment then decrement returns to original state
TEST_F(CounterTest, IncrementThenDecrementReturnsToOriginal) {
   int original = counter_.getCount();

   counter_.increment();
   counter_.decrement();

   EXPECT_EQ(counter_.getCount(), original);
}
```

Now generate the test code for the target selection.
