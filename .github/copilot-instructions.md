# Copilot Instructions (Graphitti)

This file onboards a coding agent to Graphitti. Trust these instructions and only search if something is missing or inaccurate. At the start of any code review or pull request review, make a one-line statement indicating that you have been onboarded using this file.

## Repository summary

- Graphitti is a high-performance C++17 simulator for graph-based systems, used for neuroscience and emergency communications modeling.
- Supports CPU and CUDA GPU builds, large graphs, and long-running simulations.
- Build system: CMake. Tests: Google Test plus regression simulations.

## Tech stack and validated tools

- C++17 with g++.
- CMake.
- clang-format (for style checks).
- Optional: CUDA (for GPU build), HDF5 (for HDF5 recorders), Boost Graph library.

## Project layout (high-signal paths)

- `Simulator/`: core simulator code. Main entry: `Simulator/Core/Graphitti_Main.cpp`.
- `Testing/`: unit tests and regression test configs; test runner: `Testing/RunTests.cpp`.
- `Testing/RegressionTesting/`: config files, GoodOutput, TestOutput, compare_matrices source.
- `Testing/UnitTesting/`: Google Test suites.
- `ThirdParty/`: vendored deps (log4cplus, cereal, TinyXPath, paramcontainer, googletest).
- `Tools/`: Python utilities for generating or visualizing graphs.
- `docs/`: developer and user documentation; Doxygen config in `docs/Doxygen/`.
- `build/`: CMake build output (generated). Contains `RuntimeFiles/` used at runtime.

Root files: `.clang-format`, `.github/`, `CMakeLists.txt`, `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `LICENSE`, `Simulator/`, `Testing/`, `ThirdParty/`, `Tools/`, `docs/`, `build/`, `configfiles/`, `config.h.in`.

GPU build requires CUDA and `-D ENABLE_CUDA=YES` during configure; optionally set `-D TARGET_ARCH=NN`.

## Key behavior references

- `Simulator/Core/Graphitti_Main.cpp`: initializes logging, selects log4cplus config, and calls `Core::runSimulation`.
- `Testing/RunTests.cpp`: initializes logging and executes all Google Tests.

## Style and C++ standards (strict)

- Use `.clang-format` at repo root; 3-space indentation, 100-column limit.
- Naming: camelCase; classes start uppercase, functions/vars lowercase.
- Braces: cuddled for control flow, isolated for function bodies; always use braces.
- Header guards: `#pragma once`.
- C++17: prefer `using` over `typedef`, use `const`/`constexpr`, explicit copy/move, `override`, smart pointers.

## CI and validation

- Unit and regression tests: [.github/workflows/tests.yml](workflows/tests.yml)
  - Runs `cmake ..`, `make -j`, `./tests`, then CPU regression sims and `compare_matrices`.
- Format check: [.github/workflows/format.yml](workflows/format.yml)
  - `clang-format --dry-run --Werror --style=file` excluding `ThirdParty/`, `docs/`, `Testing/lib/`.
- Docs: scheduled/manual GitHub Pages builds use Doxygen.
- PlantUML: updates diagrams on `.puml` changes.

## Contribution hygiene

- Do not work directly on `master`.
- Branch naming: `issue-####-short-description`.
- PR title: `[ISSUE-####] ...` and link issue in PR description.
