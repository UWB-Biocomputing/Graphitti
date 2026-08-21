# GitHub Actions Workflows

This page is dedicated to documentation of any automation files found within the [.github/workflows](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/workflows) folder.

## Doxygen and GitHub Pages Action gh-pages.yml

This action is triggered on a monthly schedule (at the first of every month) or manually via `workflow_dispatch` from the Actions tab. When triggered, the Doxygen documentation is regenerated and published to the `gh-pages` branch. First, it checks out the repository using [actions/checkout](https://github.com/actions/checkout). Next, the Doxygen files are regenerated using [mattnotmitt/doxygen-action](https://github.com/mattnotmitt/doxygen-action). Lastly, the `gh-pages` branch is updated with the `docs` folder and published using the [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages) action. When this is done, the branch is committed as an orphan.

### Mermaid Diagrams _config.yml
Since mermaid is set to `true` in `_config.yml`, anytime GitHub Pages action is triggered, `_includes/head-custom.html` uses JavaScript to seek for any files that contain `mermaid` (once webpage finishes loading), then loads the Mermaid library to render diagrams directly in the browser.

## Code Style Check format.yml

This action is triggered on pushes and pull requests that modify C++ source or header files (`.cpp`, `.h`). It executes `clang-format` to verify compliance with the repository style guidelines.

## Unit Tests unit-tests.yml

This action runs on pushes and pull requests (excluding documentation-only changes). It compiles the unit test binary (`make tests`) with CMake and executes `./tests` for rapid feedback on test status.

## Regression Tests regression-tests.yml

This action runs on pushes and pull requests (excluding documentation-only changes). It compiles the simulator binary (`make cgraphitti`) and the matrix verification utility (`compare_matrices`), executing all 10 simulation test configurations against reference output matrices.

## Auto-Close Merged Issues close-merged-issues.yml

This action triggers automatically whenever a pull request is merged into `SharedDevelopment` or `master`. It extracts referenced issue numbers from the PR title, branch name, and PR description (e.g. `[issue-123]`, `fixes #123`, `closes #123`, `issue-123`), checks if the issue is currently open on GitHub, and automatically closes it with a comment linking the merged pull request.

## Maintenance Scripts

### Stale Issue Cleanup cleanup_stale_issues.sh

The script [.github/scripts/cleanup_stale_issues.sh](file:///Users/stiber/GitHub/Graphitti/.github/scripts/cleanup_stale_issues.sh) scans merged pull requests on GitHub to identify referenced issues (such as `[issue-123]`, `fixes #123`, or `closes #123`) that remain in the `OPEN` state, allowing batch closing of issues resolved by merged PRs.

- **Dry Run (Preview candidate issues without modifying)**:
  ```bash
  ./.github/scripts/cleanup_stale_issues.sh --dry-run
  ```
- **Execute Issue Closure**:
  ```bash
  ./.github/scripts/cleanup_stale_issues.sh --execute
  ```
- **Options**:
  - `-d, --dry-run`: Preview candidate issues without closing them (default).
  - `-x, --execute`: Close the identified open issues with a reference to the merged pull request.
  - `-l, --limit NUM`: Maximum number of merged PRs to inspect (default: `100`).
  - `-h, --help`: Display usage help.

