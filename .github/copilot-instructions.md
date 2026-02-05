# GitHub Copilot Instructions for PR Reviews

## Acknowledgment

When reviewing pull requests or providing development assistance for the Graphitti project, **always begin your response by acknowledging that you are using the Graphitti project-specific instructions**. This confirmation helps verify that you have loaded all relevant project context, coding standards, and review guidelines.

Example acknowledgment: "I'm reviewing this using the Graphitti project guidelines and standards."

## Project Overview

**Graphitti** is a high-performance simulator of graph-based systems, primarily applied to computational neuroscience and emergency communications systems. The simulator:

- Supports vertices and edges with internal state
- Implements message passing between vertices over edges
- Allows dynamic graph architecture changes (edge creation and destruction)
- Manages vertex spatial locations (x, y coordinates, with potential z-axis support)
- Supports multiple vertex types
- Provides flexible data recording during simulations
- Runs on both CPUs and GPUs
- Can simulate very large graphs (tens of thousands of vertices, hundreds of thousands to millions of edges)
- Handles long-duration simulations (billions of time steps)

### Technology Stack

- **Language**: C++17
- **Build System**: CMake
- **Testing**: Custom testing framework with unit and regression tests
- **Dependencies**: See ThirdParty/ directory (cereal, log4cplus, paramcontainer, TinyXPath)

## Code Formatting Etiquette

> **Note**: Code is written once and read a thousand times. Hence, it is important we keep our codebase consistent to improve readability.

All C++ code **MUST** adhere to the project's formatting standards. A `.clang-format` file is provided at the repository root to automate formatting.

### File Extensions

- Use `.cpp` and `.h` for C++ code, and `.cu` for CUDA source files
- Name files with _exactly_ the same name (including capitalization) as the primary classes they define

### Indentation

- Indent using _three spaces_. **Not tabs**. Spaces.

### Naming Conventions

- Use [cC]amelCase naming, rather than underscores
- Classes start with capital letters
- Functions and variables start with lowercase letters

### Spaces

- Put spaces after list items and method parameters: `f(a, b, c)`, not `f(a,b,c)`
- Put spaces around operators: `x += 1`, not `x+=1`
- Don't put spaces after or before parentheses: `f(a)`, not `f( a )`

### Braces

- [Cuddle braces](http://blog.gskinner.com/archives/2008/11/curly_braces_to.html) for loops and conditionals (except for right braces closing a code block)
- Put isolated braces on their own lines for functions
- Always use braces even when a code block is a single line (to prevent bugs when it later expands to multiple lines)

```cpp
if (x > m) {
    x--;
} else {
    x++;
}

int f(a)
{
    return a;
}
```

### Line Length

- Limit code to **100 character line lengths**

### Condition Checks

- Use explicit checks:
  - `if (aPointerVar == nullptr)`, not `if (aPointerVar == 0)`
  - `if (!aBoolFlag)`, not `if (aBoolFlag == false)`
  - `if (aCharVar == '\0')`, not `if (aCharVar == 0)`

### Empty Lines

- Use an empty line between methods
- Use empty lines around multi-line blocks
- Use Unix end-of-line characters (`\n`)

### Header Guards

- Use `#pragma once` instead of `#define` guards

## C++ Design and Coding Standards

The project targets **C++17**. Follow these guidelines when using C++ features:

### Use of const and constexpr

- Use `const` to indicate that a value will not change once assigned
- Use `constexpr` for values known at compile time
- Put `const` and `constexpr` first: `const int i = 100;` not `int const i = 100;`
- Use `const` qualifier on class functions that don't change member variables
- Apply `const` to function parameters, methods, and non-local variables where appropriate

```cpp
class A {
    const int i = 100;
    const int * const num = &i;

    int function1(char c) const;
    const int* function2(const char* string) const;
};
```

### Copy and Move Operations

- A class must make it clear whether it is copyable, move-only, or neither by explicitly declaring and/or deleting the appropriate operations
- Use compiler options `= default` and `= delete`

```cpp
class Copyable {
 public:
  Copyable(const Copyable& other) = default;
  Copyable& operator=(const Copyable& other) = default;
};

class MoveOnly {
 public:
  MoveOnly(MoveOnly&& other) = default;
  MoveOnly& operator=(MoveOnly&& other) = default;
};

class NotCopyable {
 public:
  NotCopyable(const NotCopyable&) = delete;
  NotCopyable& operator=(const NotCopyable&) = delete;
};
```

### Smart Pointers

- Use smart pointers (`unique_ptr`, `shared_ptr`) where possible to improve readability and eliminate manual ownership bookkeeping
- Prefer `unique_ptr` to express exclusive ownership
- Avoid `shared_ptr` without a very good reason (e.g., avoiding expensive copies, immutable objects)
- **Never use `auto_ptr`**. Use `unique_ptr` instead

```cpp
std::unique_ptr<int> value1(new int(10));

// OR

std::unique_ptr<int> value2;
value2.reset(new int(47));
```

### Aliases

- Prefer `using` over `typedef` for consistency with C++ syntax and template support

### Inputs and Outputs

- Prefer to return by value or by reference. Avoid returning pointers unless they can be null
- Non-optional input parameters should usually be values or const references
- Non-optional output/input-output parameters should usually be references
- Use `optional` to represent optional by-value inputs
- Use non-const pointers for optional outputs

### Override Keyword

- Explicitly annotate overrides of virtual functions with `override`
- Do **not** use `virtual` when declaring an override
- Benefits: Catches errors at compile time and serves as documentation

### Return References from Accessor Methods

- Accessor methods (getters) should generally return references (except for primitives) to avoid unnecessary copying
- Return `const` references if the method doesn't modify data (promotes const-correctness)
- Return non-const references if the method does modify data
- Avoid returning data by address unless necessary

## PR Review Guidelines

When reviewing pull requests, ensure the following:

### 1. Code Formatting Compliance

- ✅ All code follows `.clang-format` rules (automated check should pass)
- ✅ 3-space indentation (no tabs)
- ✅ Lines do not exceed 100 characters
- ✅ [cC]amelCase naming convention followed
- ✅ Proper spacing around operators, commas, and parentheses
- ✅ Braces are cuddled for loops/conditionals, isolated for functions
- ✅ Always use braces even for single-line blocks
- ✅ File extensions are `.cpp` and `.h` (`.cu` for CUDA)
- ✅ Header guards use `#pragma once`

### 2. Branch and Issue Compliance

- ✅ PR is targeting the `development` branch, NOT `master`
- ✅ Branch name follows format: `issue-####-short-description`
- ✅ PR title starts with `[ISSUE-####]`
- ✅ Linked issue is properly referenced in the PR description
- ✅ Issue is assigned to the PR author

### 3. C++ Design Standards

- ✅ Proper use of `const` and `constexpr` (prefer first position in declarations)
- ✅ Copy/move operations explicitly declared or deleted as appropriate
- ✅ Smart pointers (`unique_ptr`, `shared_ptr`) used where appropriate; never `auto_ptr`
- ✅ `using` preferred over `typedef`
- ✅ Input parameters are values or const references; outputs are references
- ✅ Override keyword used (without `virtual`) for overridden methods
- ✅ Accessor methods return references (const or non-const) except for primitives

### 4. Code Quality

- ✅ Code is clear, maintainable, and well-documented
- ✅ No unnecessary complexity or code duplication
- ✅ Proper error handling where appropriate
- ✅ Memory management is correct (no leaks)
- ✅ Thread safety considerations for multi-threaded code

### 5. Testing Requirements

- ✅ All automated GitHub Actions tests pass
- ✅ Unit tests are included for new functionality
- ✅ Existing tests still pass
- ✅ Manual tests completed if required (especially for GPU functionality)

### 6. Documentation

- ✅ Code comments explain complex logic
- ✅ Public API changes are documented
- ✅ README or other docs updated if user-facing changes

### 7. Performance Considerations

- ✅ No obvious performance regressions
- ✅ Efficient algorithms used for graph operations
- ✅ Memory usage is reasonable for large-scale simulations

## Suggestions Format

When providing code suggestions during reviews:

1. **Always format according to `.clang-format` rules**
2. Use proper C++17 features and idioms
3. Maintain consistency with existing codebase patterns
4. Provide clear explanations for suggested changes
5. Reference specific lines using GitHub's suggestion format when possible

## Common Issues to Watch For

- 🚫 Direct commits to `master` branch
- 🚫 Code that doesn't pass automated format checks
- 🚫 Missing or inadequate test coverage
- 🚫 Undocumented API changes
- 🚫 Breaking changes without proper discussion
- 🚫 Memory leaks or unsafe memory operations
- 🚫 Hard-coded values that should be configurable
- 🚫 Overly long functions (consider refactoring)

## References

- [Code Formatting Etiquette](https://uwb-biocomputing.github.io/Graphitti/Developer/codingConventions.html)
- [C++ Design and Coding Standards](https://uwb-biocomputing.github.io/Graphitti/Developer/cppStyleGuide.html)
- [GitFlow Documentation](../docs/Developer/GitFlow.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)

---

**Remember**: The goal is to maintain high code quality while being constructive and supportive of contributors. All suggestions should help improve the codebase while respecting the effort put in by the contributor.
