# Contributing to The GRAPHITTI Project

Thank you for your interest in the Graphitti Project, which includes the Workbench software and data provenance system. This project operates with an [Apache 2.0 license](../../LICENSE) which provides wide reusability and adaptability, under the proviso of citing the originators and maintaining provenance information.

## External from the BioComputing Lab
For people outside of the [UW Bothell Biocomputing laboratory](http://depts.washington.edu/biocomp/) (BCL), we use a [fork and pull development model](https://help.github.com/articles/about-collaborative-development-models/). If you're interested in adapting this project for your own use, then please feel free to make your own copy of this repository and adapt it to your work. We would be greatly interested to learn about what you do, potentially incorporating your work back into this main repository. *Please cite us in your work*; the repository [README](../../Desktop/Graphitti/BG-reorg/README.md) has a DOI for that purpose.

## For the BioComputing Lab
For UW Bothell students interested in working in the BCL, we use a [shared repository development model](https://help.github.com/articles/about-collaborative-development-models/). If you're interested in contributing directly to this project, then please contact [Prof. Michael Stiber](mailto:stiber@uw.edu) and read the information below.

## GitHub Workflow

- Please read up on Github basics (including [Managing your work on GitHub](https://help.github.com/categories/managing-your-work-on-github/)).
- Seek the guidance of more senior lab members regarding how to get started. 
- Please ***DO NOT WORK DIRECTLY ON THE MASTER BRANCH*** (yes, there are exceptions, but they are few and far between). 
- Instead, create a branch, do what you intend, *check that your haven't broken anything*, and then merge your branch into master. If you're unsure about doing such a merge, then discuss what you've done at a lab meeting or open a pull request (read more about [pull requests](http://help.github.com/pull-requests/)).

If you're creating a branch that is in response to an issue, then name the branch accordingly, i.e., "issue-3141". This implies a one-to-one correspondence between issues and branches. If you want to work on an issue and it seems pretty clear that it's a big undertaking, then talk with the group. Possibly, it will be a branch that exists for a while, and you may need to merge the master branch back into it multiple times as you work on it. But, it's also possible that the issue in question should really be broken into sub-issues that can be worked on separately. You can use [the GitHub syntax](https://help.github.com/articles/closing-issues-using-keywords/) to close issues directly from commits or pull requests upon merge into the master branch.

We are working on developing a Jenkins server to help validate changes, so that you'll more easily know whether what you've done passes all of our tests for correctly working (or, more pedantically, behaving in a manner consistent with the current release version). More on this to come later.

*Please document what you've done*, not only in your commit messages but also with useful comments in code and via changes to the github pages content in the docs directory.

todo: link to codingConventions.md

