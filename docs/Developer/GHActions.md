# GitHub Actions Workflows

This page is dedicated to documentation of any automation files found within the [.github/workflows](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/workflows) folder.

## Doxygen and GitHub Pages Action gh-pages.yml

This action is triggered on a monthly schedule (at the first of every month) or manually via `workflow_dispatch` from the Actions tab. When triggered, the Doxygen documentation is regenerated and published to the `gh-pages` branch. First, it checks out the repository using [actions/checkout](https://github.com/actions/checkout). Next, the Doxygen files are regenerated using [mattnotmitt/doxygen-action](https://github.com/mattnotmitt/doxygen-action). Lastly, the `gh-pages` branch is updated with the `docs` folder and published using the [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages) action. When this is done, the branch is committed as an orphan.

### Mermaid Diagrams _config.yml
Since mermaid is set to `true` in `_config.yml`, anytime GitHub Pages action is triggered, `_includes/head-custom.html` uses JavaScript to seek for any files that contain `mermaid` (once webpage finishes loading), then loads the Mermaid library to render diagrams directly in the browser.

## Code Style Check format.yml

This action is triggered on pushes to `master` and pull requests that modify C++ source or header files (`.cpp`, `.h`). It executes `clang-format` to verify compliance with the repository style guidelines.

## Unit and Regression Tests tests.yml

This action runs on pushes and pull requests (excluding documentation-only changes). It compiles the simulator with CMake, runs unit tests (`./tests`), and executes regression test configurations against reference output matrices using `compare_matrices`.
