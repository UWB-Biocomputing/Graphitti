# Contributing to The Graphitti Project

Thank you for your interest in the Graphitti Project, which includes the Workbench software and data provenance system. This project operates with an [Apache 2.0 license](LICENSE) which provides wide reusability and adaptability, under the proviso of citing the originators and maintaining provenance information.

## External from the Intelligent Networks Lab
For people outside of the [UW Bothell Intelligent Networks laboratory](http://depts.washington.edu/biocomp/) (INL), we use a [fork and pull development model](https://help.github.com/articles/about-collaborative-development-models/). If you're interested in adapting this project for your own use, then please feel free to make your own copy of this repository and adapt it to your work. We would be greatly interested to learn about what you do, potentially incorporating your work back into this main repository. *Please cite us in your work*; the repository [README](README.md) has a DOI for that purpose.

If you're making modifications that you'd like to be merged into our code base, then please see the Workflow section, below. When unsure, contact us ahead of time.

## For the Intelligent Networks Lab
For UW Bothell students interested in working in the BCL, we use a [shared repository development model](https://help.github.com/articles/about-collaborative-development-models/). If you're interested in contributing directly to this project, then please contact [Prof. Michael Stiber](mailto:stiber@uw.edu) and read the information below.

## Workflow

- Please read up on Github basics (including [GitHub Issues](https://help.github.com/categories/managing-your-work-on-github/)).
- Seek the guidance of more senior lab members regarding how to get started. 
- Please ***DO NOT WORK DIRECTLY ON THE MASTER BRANCH*** (yes, there are exceptions, but they are few and far between). 
- Instead, please follow the lab workflow that follows.

0. Review our [Coding Conventions](https://uwb-biocomputing.github.io/Graphitti/Developer/codingConventions.html). Your work will be rejected if it doesn't conform.

1. Your work should be in response to one or more issues. If you are planning to work on something that is a small part of an existing issue, then likely that issue is a placeholder "umbrella" that was generated in lieu of thinking through all related details. In that case, now is the time for you to think it through and break that issue down into actionable items —— new issues that partially or completely replace the umbrella.

2. The issue(s) you are working on should be part of a project for the current quarter. If not, make it so. Assign yourself to those issue(s) and move them from "To do" to "In progress".

3. Create a new feature branch for your work. If the branch is in response to a single issue, then you can name the branch accordingly, i.e., "issue-3141"; otherwise, just give it a logical name. Add a comment to the issue(s) including a link to the feature branch.

4. Make changes to the feature branch (commit/push).

5. Create a pull request for your branch (read more about [pull requests](http://help.github.com/pull-requests/)). You may choose to do this early in your work on the branch or later. Link the issue(s) you're working on to this pull request.

6. Before requesting a review of your pull request, *check that your haven't broken anything*. This means checking that all of our automated GitHub actions have passed their tests (this will show directly in the pull request) and that any required manual tests have passed (some of our tests take a while to run, so rather than do them for every commit to a pull request, we just run them manually when such requests are close to done).

7. Request a review of your pull request. If you have a designated reviewer, ask that person; otherwise, ask Prof. Stiber or ask during a lab meeting who should review your pull request. You should see the linked issue(s) automatically move to "Review in progress" in the project when you request the review, and to "Reviewer approved" when approved.

8. When your pull request is approved, you can merge it. You should see that the relevant issues automatically move to "Done" in the project.

9. Once you've verified that the merge is done, you can delete your feature branch.

***Please document what you've done***, not only in your commit messages but also with useful comments in code and via changes to the github pages content in the docs directory.

