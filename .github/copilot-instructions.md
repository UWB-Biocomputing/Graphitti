# Graphitti Copilot Instructions

## 1. Project Overview

Graphitti is a high-performance graph-based simulator for computational neuroscience and emergency communications research, developed at the University of Washington Bothell. It simulates large-scale graphs (tens of thousands of vertices; millions of edges) over billions of time steps. It runs on both CPUs and GPUs.

Repository: https://github.com/UWB-Biocomputing/Graphitti

## 2. Tech Stack

- **Language:** C++17 (strict)
- **Build System:** CMake
- **Testing:** Google Test (gtest)
- **Logging:** log4cplus
- **GPU:** CUDA (guarded by `USE_GPU` and `ENABLE_CUDA` macros)
- **Parallelism:** CUDA (GPU); OpenMP planned for CPU multi-threading
- **Data Recording:** HDF5 (binary) and XML
- **Config Format:** XML (parsed via `ParameterManager`)
- **OS:** GNU/Linux

## 3. Code Standards

Apply these rules to every code generation and review task.

### Language & Modern C++

- **Standard:** C++17. Do not use features from C++20 or later.
- Avoid manual `delete` and owning raw pointers; prefer RAII and smart pointers (`std::unique_ptr` / `std::shared_ptr`).
- Avoid `printf`; use standard streams or log4cplus.
- Prefer `<algorithm>` when it improves clarity, but traditional loops are acceptable in performance-critical paths.
- Use `[[nodiscard]]` on functions with non-void return values to prevent silent discard of error codes or computed results.
- Use `const` and `constexpr` wherever possible.
- Use `#pragma once` for all headers.
- Use explicit `override` on virtual function overrides.

### Formatting

- **Indentation:** 3 spaces. Not 2, not 4. This is a project-wide convention for codebase consistency.
- **Column Limit:** 100 characters.
- **Naming:**
  - `CamelCase` for classes: `Vertex`, `Graph`, `EdgeIndexMap`.
  - `camelCase` for variables and functions: `numVertices`, `calculateEdges`.
  - No `snake_case`.
- **Braces:**
  - Control flow: Cuddled (`} else {`).
  - Functions: Opening `{` on a new line.
  - Always use braces, even for single-line blocks, to prevent dangling-else bugs when lines are added during maintenance.

## 4. Architecture Map

Understand where code belongs so you can suggest correct file paths and appropriate performance considerations.

- **`Simulator/Core/`** — The simulation hot path. Code here must be highly optimized. `Graphitti_Main.cpp` is the entry point; `Simulator::simulate()` and `Simulator::advanceEpoch()` are the main loop.
- **`Simulator/Edges/`** and **`Simulator/Vertices/`** — Graph element implementations with internal state. Frequently called per time step.
- **`Simulator/Recorders/`** — Data recording subsystem. Supports `XmlRecorder` and `HDF5Recorder`.
- **`Testing/UnitTesting/`** — Google Test unit tests. Must be fast and isolated.
- **`Testing/RegressionTesting/`** — Full simulation runs. Only modify when physics or logic changes.
- **`ThirdParty/`** — External dependencies. **Read-only.** Do not suggest changes here.

## 5. Pull Request Review Priorities

When reviewing PRs or suggesting fixes, check in this order:

1. Flag unnecessary object copying; suggest `const&` or move semantics.
2. Flag expensive allocations or dynamic_cast calls inside simulation loops (`Simulator/Core/`).
3. Identify potential cache misses in hot loops.
4. Check for iterator invalidation and thread-safety in shared data structures (OpenMP/CUDA context).
5. Verify `CMakeLists.txt` is updated if source files were added or removed.
6. Verify all required headers are included.

## 6. Testing Requirements

- **New logic** must have corresponding `TEST()` or `TEST_F()` cases in `Testing/UnitTesting/`.
- **Bug fixes** require a regression test if the bug was a logic error.
- **GPU code** (`.cu` files) must check `ENABLE_CUDA` macros and have CPU-path equivalents tested.
- Test names use `PascalCase`: `TEST(Graph, AddsVertexCorrectly)`.
- Use `EXPECT_*` for non-fatal assertions; use `ASSERT_*` for preconditions where continuing would crash.

## 7. Interaction Behavior

- **On PR review:** Start by identifying the impact area (e.g., "This PR modifies the core simulation loop; checking performance constraints...").
- **On code generation:** Always specify the file path where the generated code should be placed, based on the Architecture Map above.
- **On refactoring:** Preserve existing public API signatures unless the user explicitly requests breaking changes.
