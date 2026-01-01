# Developer Documentation

If you're developing Graphitti code, then here are your reference documents. 

Writing new code? Then make sure to follow our [contributing guide] and  *document your code here*. 

Reading code that isn't obvious? When you figure out how it works, then *document it here* and *document it in comments in the code.*

## Student Quick Start

Students, use this [quickstart guide](StudentSetup.md) to help setup, use, and develop with Graphitti.

## Software Development Process

- To further understand our development process, please check out our [gitflow documentation](GitFlow.md).

- Your pull requests will not be approved if you do not adhere to our [coding conventions](codingConventions.md).

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
    - Here is a overview [block UML diagram](ClassDiagrams/hand-drawn.pdf)
    - Here is a list of [UML class diagrams](classDiagrams.md) of Graphitti
    - Here are the [sequence UML diagrams](sequenceDiagrams.md) for the Graphitti system
- Doxygen
    - Documentation generated from source code
    - Doxygen provides web-based indices and hierarchical views of Graphitti's class and file structures
    - [Visit Doxygen Generated Documentation]
    - Document code in the `.h` file using the [Doxygen Style Guide](../Doxygen/DoxygenStyleGuide.md) format
    - [Doxygen Update Guide](../Doxygen/DoxygenUpdateGuide.md)
- [Event buffering](eventBuffering.md) in vertex classes.
- [Performing Analyses](PerformingAnalyses.md)
- [Neuro Implementation](NeuroImplementation.md)
- [GraphManager and InputManager classes](GraphAndEventInputs.md)
- [Configuration](../User/configuration.md)


---------
[<< Go back to the Graphitti home page](../index.md)

[//]: # (Moving URL links to the bottom of the document for ease of updating - LS)
[//]: # (Links to repo items which exist outside of the docs folder need an absolute link.)

[contributing guide]: <https://github.com/UWB-Biocomputing/Graphitti/blob/master/CONTRIBUTING.md>
[Visit Doxygen Generated Documentation]: <https://uwb-biocomputing.github.io/Graphitti/Doxygen/html/index.html>
   