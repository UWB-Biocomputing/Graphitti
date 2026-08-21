# GitHub Actions Workflows

This page is dedicated to documentation of any automation files found within the [.github/workflows](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/workflows) folder.

## Doxygen and GitHub Pages Action gh-pages.yml

This action is triggered on a monthly schedule (at the first of every month) or manually via `workflow_dispatch` from the Actions tab. When triggered, the Doxygen documentation is regenerated and published to the `gh-pages` branch. First, it checks out the repository using [actions/checkout](https://github.com/actions/checkout). Next, the Doxygen files are regenerated using [mattnotmitt/doxygen-action](https://github.com/mattnotmitt/doxygen-action). Lastly, the `gh-pages` branch is updated with the `docs` folder and published using the [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages) action. When this is done, the branch is committed as an orphan.

### Mermaid Diagrams _config.yml
Since mermaid is set to `true` in _config.yml, anytime GitHub Pages action is triggered, _includes/head-custom.html uses Javascript to seek for any files that contains `mermaid` (once webpage finishes loading), then loads mermaid library.

## PlantUML Action plantUML.yml

The plantUML action occurs anytime a plantUML file is modified or added during a pull request or a push to the master branch. These .puml files are supposed to be located in the UML folder within the Developer folder. This action starts by checking out the repository using [actions/checkout](https://github.com/actions/checkout) with a fetch depth of 0. The next step is to grab all of the .puml files that need to be turned into images. This is done by using a basic bash command to grab all .puml files which is then piped into an awk script to parse out the unnecessary files and construct an output string with all the necessary files. The output string will look like so: "file1.puml file2.puml file3.puml file4.puml\n". This output string is then confirmed by an echo command which prints out the string to the actions terminal. Next, the .png and .svg files are generated from the .puml files in the output string using a fork of [holowinski/plantuml-github-action]. These files are placed within the diagrams folder located within the UML folder. Lastly, the local changes are committed then pushed to the remote repository using [stefanzweifel/git-auto-commit-action](https://github.com/stefanzweifel/git-auto-commit-action).


[//]: # (Moving URL links to the bottom of the document for ease of updating - LS)
[//]: # (Links to repo items which exist outside of the docs folder need an absolute link.)

[holowinski/plantuml-github-action]: <https://github.com/UWB-Biocomputing/plantuml-github-action>
