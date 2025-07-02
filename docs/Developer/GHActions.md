# GitHub Actions Workflows

This page is dedicated to documentation of any automation files found within the [.github/workflows](https://github.com/UWB-Biocomputing/Graphitti/tree/master/.github/workflows) folder.

## Doxygen and GitHub Pages Action gh-pages.yml

This action is Triggered on a monthly schedule. At the first of every month the doxygen documentation will be regenerated so that any new changes will be updated to the GitHub pages. First, it checks-out the repository using [actions/checkout](https://github.com/actions/checkout). Next, the doxygen files are regenerated using [mattnotmitt/doxygen-action](https://github.com/mattnotmitt/doxygen-action). Lastly, the gh-pages branch is updated with the new docs folder and published using the [peaceiris/actions-gh-pages](https://github.com/peaceiris/actions-gh-pages) action. When this is done, the branch is committed as an orphan to keep the branch as clean as possible.

## Manual GitHub Pages Action publish-gh-pages.yml

The manual GitHub Pages action is a feature that came from wanting to quickly publish changes to documentation files in our docs folder. This action is activated by navigating to the actions tab, selecting the "Publish GitHub Pages Manually" workflow, then toggling the run workflow button on the desired branch. The branch to run this action on will typically be the master branch as that is the one with the most up to date documentation. Once toggled, it will take the docs files from the selected branch and publish them to the gh-pages branch as a forced orphan just like in the gh-pages.yml workflow. This action will also regenerate the Doxygen files in the same way the gh-pages.yml script does. This is done because other branches doesn't hold the Doxygen/html files and would lose this information if not regenerated during this script.

## PlantUML Action plantUML.yml

The plantUML action occurs anytime a plantUML file is modified or added during a pull request or a push to the master branch. These .puml files are supposed to be located in the UML folder within the Developer folder. This action starts by checking out the repository using [actions/checkout](https://github.com/actions/checkout) with a fetch depth of 0. The next step is to grab all of the .puml files that need to be turned into images. This is done by using a basic bash command to grab all .puml files which is then piped into an awk script to parse out the unnecessary files and construct an output string with all the necessary files. The output string will look like so: "file1.puml file2.puml file3.puml file4.puml\n". This output string is then confirmed by an echo command which prints out the string to the actions terminal. Next, the .png and .svg files are generated from the .puml files in the output string using [cloudbees/plantuml-github-action](https://github.com/cloudbees/plantuml-github-action). These files are placed within the diagrams folder located within the UML folder. Lastly, the local changes are committed then pushed to the remote repository using [stefanzweifel/git-auto-commit-action](https://github.com/stefanzweifel/git-auto-commit-action).

