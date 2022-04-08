.. _contribguide:

Contribution Guide
==================

Bug Reporting
-------------
All bugs or general issues can be reported on the Github issues page:
https://github.com/mradaideh/neorl/issues

Getting Started
---------------
In order to contribute code that may be included in the master NEORL distribution, the following steps can be taken.
In general, all contributions should be submitted as a pull request to the master NEORL repository.
Specifically, the steps provided below can be used to incorporate code into the NEORL repository.

1. Create a fork of the master NEORL repository. Through github, this can be done by selecting the "fork" button on the 
   NEORL master page found at: https://github.com/mradaideh/neorl

.. image:: ../images/fork_button_red.png
   :scale: 70 %
   :alt: alternate text
   :align: left

2. Once this is done, a new repository will be created under your ownership. It will exist under `github_username/neorl`.
   From here, the forked repository can be cloned like any other repository.
   Navigate to a directory you plan on working in and enter the command::
   git clone git@github.com:github-un/neorl.git
   
   Replace the `github-un` with your github username

3. From here, the forked repository can be comitted to using typical `git` practices.

Incorporating Changes in master Repository
----------------------------------------
Often when making large contributions, it may be necessary to work on the fork for an extended period of time where
 updates have been pushed to the upstream (`mradaideh/neorl`) repository after you originally created the fork.
 In order to incorporate these new updates into your local fork:
1. Commit and push all work in the local repository.
2. First add the upstream repository into known remote repositories::
   git remote add upstream https://github.com/mradaideh/neorl
3. Fetch changes that have been made to upstream repository::
   git fetch upstream
4. Merge changes made to upstream repository into local repository::
   git merge upstream/master master
5. Push from local repository to remote::
   git push

Submitting a Pull Request
-------------------------
Once a major contribution to the NEORL code base has been developed and is ready to be incorporated into 
the upstream repository (`mradaideh/neorl`), a pull request can be submitted.
To do this::
1. While logged into the account with the forked repository containing the changes, navigate to the github 
   page for the upstream NEORL repository: https://github.com/mradaideh/neorl
2. Select the "Pull requests" tab near the top of the page and select "new pull request"

.. image:: ../images/annotate_prerequest.png
   :scale: 70 %
   :alt: alternate text
   :align: left

3. Select "compare across forks" link
.. image:: ../images/compare_across.png
   :scale: 70 %
   :alt: alternate text
   :align: left
4. From here, the "base repository" should be set to `mradaideh/neorl/master` and the "head repository"
   should point to the repository that is under your name.
5. Press the "Create pull request" button and fill out the submission fields and submit it for review!




