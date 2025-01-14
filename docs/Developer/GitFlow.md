# INL Software Development Workflow

Our workflow is based on GitFlow, with both shared and personal development branches, release branches, and hotfix branches.

## What is GitFlow?

GitFlow is a branching model for Git, created by [Vincent Driessen.](https://nvie.com/posts/a-successful-git-branching-model/) 

## Key Features

### Parallel Development

One of the great things about GitFlow is that it makes parallel development very easy, by isolating new development from finished work. New development (such as features and non-emergency bug fixes) is done in feature branches and is only merged back into the development branch after going through extensive unit/regressional testing and review through the pull request mechanism.

### Collaboration

Feature branches also make it easier for developers to collaborate on the same feature, because each feature branch is essentially a sandbox where you isolate and test the changes that are only necessary to get a new feature working, making it clear what each collaborator is working on.

### Conflict Resolution

Developers who are working on code that will take an extended period of time to complete (likely composed of multiple issues/features) are able to maintain their personal development branches with code that may break others'. Each developer is responsible for resolving conflicts introduced by others' code before merging into the shared development branch.

### Release Staging Area

As new development is completed, it gets merged back into the development branch, which is a staging area for all completed features that haven't yet been released. So when the next release is branched off of development, it will automatically contain all of the new tasks that have been finished.

## How it Works

***ALL OF OUR DIAGRAMS READ FROM TOP TO BOTTOM***

The diagram below provides a high-level view of our workflow overall operation. The main points are:

* Shared and personal development branches are used.
* With the exception of hotfixes (described later and not shown in this diagram), the "master" branch is only updated when a release is to be made.
* Releases are made through creation of "release" branches. Release branches are used to resolve all conflicts and complete all testing before merging into "master" and producing a tagged release.
* Feature branches are associated with individual issues. They are merged into personal development branches.
* Users periodically check the shared development branch for changes since the last divergence of their personal development branch (i.e., how many commits behind are they). If there are any commits in the shared development branch that aren't in their personal one, they merge shared development into their personal branch. This should be done no less frequently than once every two weeks; better that it is done weekly.
* The above check ***must*** be done before merging a personal development branch into the shared development branch.
* Each user is responsible for resolving any conflicts or broken code resulting from the merge of shared development into their personal development branch. Of course, they can seek help from the code author.
* Basically, the personal development code must be working and passing all tests before it is merged into shared development. Since is should be up-to-date with shared development, the same will be true for shared development after a merge.
* Generally, after being merged into shared development, the personal development branch may be deleted.
* So, from that point of view, a personal development branch is like a "super feature branch", resolving multiple related issues that individually can't be merged into shared development because they don't leave the code in a "good" (or even working) state.
* The names shown below are mostly not what we actually call these branches; see below and our onboarding guide for naming conventions.

### High Level Overview of Our Process

```mermaid
%%{init: { 'logLevel': 'debug', 'theme': 'base', 'themeVariables': {
    'git0': '#2E9AFE',
    'gitInv0': '#2E9AFE',
    'git1': '#A829FF',
    'git2': '#FFBF00',
    'git3': '#8FED0A',
    'git4': '#A4A4A4',
    'git5': '#8FED0A',
    'git6': '#A4A4A4',
    'tagLabelFontSize': '12px'
},'gitGraph': {'rotateCommitLabel': true, 'showBranches': true, 'showCommitLabel':false, 'mainBranchName': 'Master'}} }%%
gitGraph TB:
   commit
   branch SharedDev order: 2
   checkout Master
   checkout SharedDev
   commit
   branch AUserDev order: 3
   checkout AUserDev
   commit
      branch FeatureA order: 4
      checkout SharedDev
      branch BUserDev order: 5
         commit
         branch FeatureB order: 6
   commit
      checkout FeatureA
      commit
      commit
   checkout AUserDev
   merge FeatureA
   checkout SharedDev
   merge AUserDev
   checkout BUserDev
   merge SharedDev
   commit  tag: "fix broken code"
   merge FeatureB
   checkout SharedDev
   merge BUserDev
   checkout SharedDev
   checkout Master
   branch Release order: 1
   merge SharedDev
   commit
   checkout Master
   merge Release type: HIGHLIGHT tag: "v1.1.2"
```


### Feature branches

Our shared development branch is considered to be the main branch, in terms of being the latest functioning code (which may or may not be ready for merging into master and creating a release). So, all personal development branches start off as functioning code and are not merged back into the shared development branch until they are once again functioning code. Until then, features are branched off of the personal development and merged back into personal development.

```mermaid
%%{init: { 'logLevel': 'debug', 'theme': 'base', 'themeVariables': {
    'git0': '#FFBF00',
    'git1': '#A4A4A4',
    'git2': '#A4A4A4'
},'gitGraph': {'rotateCommitLabel': true, 'showBranches': true, 'showCommitLabel':true, 'mainBranchName': 'AUserDev'}} }%%
   gitGraph TB:
      commit id: "initial commit NG911"
      branch FeatureA
      commit id: "[ISSUE-412] Name Of Issue" tag:"pull-request"
      commit id: "change hardcodeded values"
      commit id: "fix typos" tag: "GitHub Actions"
      commit id: "merge ready" type: HIGHLIGHT tag:"Review"
      checkout AUserDev
      commit id: "initial commit project 2"
      branch FeatureB
      checkout AUserDev
      merge FeatureA tag: "merged featureA"
      checkout AUserDev
      checkout FeatureB
      commit id: "[ISSUE-143] Name Of Issue" tag:"PR"
      commit id: "merge ready " type: HIGHLIGHT tag:"Review"
      checkout AUserDev
      merge FeatureB tag: "merged featureB"
```

### Merging to Master Branch
The master and development branches exist parallel to one another. We consider the development branch to be the main branch where the source code always reflects a state with the latest delivered development changes. Once the development branch is ready to merge back to the master, we create a release branch (not supported in our document). Our version can either cherry-pick the developments we want into the master or revert the changes and merge to the master and re-revert the changes (not supported in the document). 

[![](https://mermaid.ink/img/pako:eNqNVNFumzAU_RXLU8QLjQIhAfyWrEkaKe2kpevDxh4cuCFWACNj2tGIf58xZSNr1xTEg8-99_ici69POOQRYIIHgxPLmCTohIyExxt4hMQgyIhgV8aGiQx5gBQaZEcL-AM8UMHoLoFCRU5BhtRjxEyOmsRP9sKfLReG-Re3NL5czpejUR-3NT5zmrePj_-DO308yGoVauCVoPmhUWIILqmEzzxNmdzQnfYiRQlKd3HgT3NBs_CgVffQs_Q9TQowjZSyrM2-o63_W1pIEEZdo3owCLJuW3Q_J1pgqFkQiwgK8AOIgvEMWUMrwDq802Toumkwz1PIZFt1gPDIS_k60KdrfhGjSQferXzrH94lUFkKmL2u_bHebr8trhzL_okaL-jLHq2LooQXgrPkFEQMSACNqgAjWeVA0M16dbNR3z2SNCYB_gqPDJ666g_p14RIcsWcgDpHb2qfv0_YSut89vVGaN-5vySqC5xv2Up9yy3q7KJL1Gfy5h-U0Z6pnoguLER1lbPw-H7_LjQfm01_1EmO1KDrKQ2wnt4AN6QRFceGqFZ5tJR8W2UhJnoucJlHao6uGY0FTfHLUGCImOTitr059AVi4pxm3zlPu0K1xOSEf2EynQxdb2pPHNf3PGti-yauMBk7Q8udup5l-QqaeHZt4mddPxr6jj9yXcd2x55rj91J_Rvb4n0O?type=png)](https://mermaid.live/edit#pako:eNqNVNFumzAU_RXLU8QLjQIhAfyWrEkaKe2kpevDxh4cuCFWACNj2tGIf58xZSNr1xTEg8-99_ici69POOQRYIIHgxPLmCTohIyExxt4hMQgyIhgV8aGiQx5gBQaZEcL-AM8UMHoLoFCRU5BhtRjxEyOmsRP9sKfLReG-Re3NL5czpejUR-3NT5zmrePj_-DO308yGoVauCVoPmhUWIILqmEzzxNmdzQnfYiRQlKd3HgT3NBs_CgVffQs_Q9TQowjZSyrM2-o63_W1pIEEZdo3owCLJuW3Q_J1pgqFkQiwgK8AOIgvEMWUMrwDq802Toumkwz1PIZFt1gPDIS_k60KdrfhGjSQferXzrH94lUFkKmL2u_bHebr8trhzL_okaL-jLHq2LooQXgrPkFEQMSACNqgAjWeVA0M16dbNR3z2SNCYB_gqPDJ666g_p14RIcsWcgDpHb2qfv0_YSut89vVGaN-5vySqC5xv2Up9yy3q7KJL1Gfy5h-U0Z6pnoguLER1lbPw-H7_LjQfm01_1EmO1KDrKQ2wnt4AN6QRFceGqFZ5tJR8W2UhJnoucJlHao6uGY0FTfHLUGCImOTitr059AVi4pxm3zlPu0K1xOSEf2EynQxdb2pPHNf3PGti-yauMBk7Q8udup5l-QqaeHZt4mddPxr6jj9yXcd2x55rj91J_Rvb4n0O)

## Detailed Run-Through of Making a Release

The basic idea is to create a release branch off of `master`, then cherry pick the commits we want to incorporate into this release. So, first of all, we want to get a list of all of the commits in `development` that aren't in `master`:

    git log --no-merges development ^master > /tmp/nomerges.txt

You'll note that we exclude merge commits above. We then go through that file and delete the commits we don't want to incorporate into our release. Next, we extract just the lines that contain the commit IDs:

    grep commit /tmp/nomerges.txt > /tmp/commits.txt

These are in reverse order (newest to oldest); we'd like the list to be oldest to newest:

    tail -r /tmp/commits.txt > /tmp/revcommits.txt

And then we can use simple find/replace in a text editor (for example, in emacs) to create a shell script with lines that look like:

    git cherry-pick -x -X theirs f622b2c36f0f29472f366470cc2d054149ce4258

In principle, if our current branch is the release branch, this should induce git to cherry pick each commit into that branch, resolving any conflicts in favor of the incoming commit. We can test by copying and pasting a line or two from that file, then, if satisfied, just `sh /tmp/revcommits.txt`.


---------
[<< Go back to the Graphitti home page](../index.md)

[<< Go back to the CONTRIBUTING.md](https://github.com/UWB-Biocomputing/Graphitti/blob/master/CONTRIBUTING.md)
