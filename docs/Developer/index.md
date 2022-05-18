# Developer Documentation

If you're developing Graphitti code, then here are your reference documents. 

Writing new code? Then make sure to follow our [contributing guide](../../CONTRIBUTING.md) and  *document your code here*. 

Reading code that isn't obvious? When you figure out how it works, then *document it here* and *document it in comments in the code.*

## Student Quick Start

Students, use this [Quickstart guide](StudentSetup.md) to help setup, use, and develop with Graphitti.

## Coding Conventions

Please adhere to our [coding conventions](codingConventions.md). Your pull requests will not be approved if you don't.

## Graphitti Repository Tools and Workflows

- CMake
    - Refer to the [CMake](CMake.md) documentation to help with any related CMake questions
- clang-format
    - Refer to the [clang-format documentation](codingConventions.md#clang-format) to help with using this tool
- GitHub Pages
    - Refer to the [GitHub Pages documentation](GHPages.md) section for an overview of how we use GitHub Pages and editing practices
- GitHub Actions Workflows
    - We have a [Doxygen Action](GHActions.md#doxygen-action) to regenerate the Doxygen documentation automatically
    - The [GitHub Pages Action](GHActions.md#github-pages-action) is another action ran along with the Doxygen one
    - Here is our [plantUML Diagrams Action](GHActions.md#plantuml-action) that regenerates our UML image documents

## Graphitti System Documentation

- Diagrams
    - Here is a overview [block UML diagram](UML/hand-drawn.pdf)
    - Here are the [sequence UML diagrams](sequenceDiagrams.md) for the Graphitti system
- Doxygen
    - Documentation generated from source code
    - Doxygen provides web-based indices and hierarchical views of Graphitti's class and file structures
    - [Visit Doxygen Generated Documentation](https://uwb-biocomputing.github.io/Graphitti/Doxygen/html/index.html)
    - Document code in the `.h` file using the [Doxygen Style Guide](../Doxygen/DoxygenStyleGuide.md) format
    - [Doxygen Update Guide](../Doxygen/DoxygenUpdateGuide.md)
- [Event buffering](eventBuffering.md) in vertex classes.
- [Performing Analyses](PerformingAnalyses.md)
- [Neuro Implementation](NeuroImplementation.md)


---------
[<< Go back to the Graphitti home page](../index.md)
