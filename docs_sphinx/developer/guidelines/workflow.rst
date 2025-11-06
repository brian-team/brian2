Development workflow
====================

Brian development is done in a `git`_ repository on `github`_. Continuous
integration testing is provided by `GitHub Actions`_, code coverage is measured with
`coveralls.io`_.

.. _git: https://git-scm.com/
.. _github: https://github.com/
.. _`GitHub Actions`: https://github.com/features/actions
.. _`coveralls.io`: https://coveralls.io/

The repository structure
------------------------
Brian's repository structure is very simple, as we are normally not supporting
older versions with bugfixes or other complicated things. The *master* branch of
the repository is the basis for releases, a release is nothing more than adding
a tag to the branch, creating the tarball, etc. The *master* branch should
always be in a deployable state, i.e. one should be able to use it as the base
for everyday work without worrying about random breakages due to updates. To
ensure this, no commit ever goes into the *master* branch without passing the
test suite before (see below). The only exception to this rule is if a commit
not touches any code files, e.g. additions to the README file or to the
documentation (but even in this case, care should be taken that the
documentation is still built correctly).

For every feature that a developer works on, a new branch should be opened
(normally based on the *master* branch), with a descriptive name (e.g.
``add-numba-support``). For developers that are members of "brian-team", the
branch should ideally be created in the main repository. This way, one can
easily get an overview over what the "core team" is currently working on.
Developers who are not members of the team should fork the repository and work
in their own repository (if working on multiple issues/features, also using
branches).

Implementing a feature/fixing a bug
-----------------------------------
Every new feature or bug fix should be done in a dedicated branch and have
an issue in the issue database. For bugs, it is important to not only fix the
bug but also to introduce a new test case (see :doc:`testing`) that makes sure
that the bug will not ever be reintroduced by other changes. It is often a good
idea to first define the test cases (that should fail) and then work on the fix
so that the tests pass. As soon as the feature/fix is complete *or* as soon as
specific feedback on the code is needed, open a "pull request" to merge the
changes from your branch into *master*. In this pull request, others can comment
on the code and make suggestions for improvements. New commits to the respective
branch automatically appear in the pull request which makes it a great tool for
iterative code review. Even more useful, GitHub Actions will automatically run the test
suite on the result of the merge. As a reviewer, always wait for the result of
this test (it can take up to 30 minutes or so until it appears) before doing
the merge and never merge when a test fails. As soon as the reviewer (someone
from the core team and not the author of the feature/fix) decides that the
branch is ready to merge, he/she can merge the pull request and optionally
delete the corresponding branch (but it will be hidden by default, anyway).

Useful links
------------
* The Brian repository: https://github.com/brian-team/brian2
* GitHub Actions tests for Brian: https://github.com/brian-team/brian2/actions
* Code Coverage for Brian: https://coveralls.io/github/brian-team/brian2
* The Pro Git book: https://git-scm.com/book/en/v2
* github's documentation on pull requests: https://help.github.com/articles/using-pull-requests
