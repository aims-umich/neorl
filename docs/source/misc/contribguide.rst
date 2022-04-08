.. _contribguide:

Contribution Guide
==================

Bug Reporting
-------------
All bugs or general issues can be reported on the Github issues page:
https://github.com/mradaideh/neorl/issues

Getting Started
---------------
In order to contribute code that may be included in the main NEORL distribution, the following steps can be taken. In general, all contributions should be submitted as a pull request to the main NEORL repository. Specifically, the steps provided below can be used to incorporate code into the NEORL repository.

#. Create a fork of the main NEORL repository. Through Github, this can be done by selecting the "fork" button on the NEORL page found at: https://github.com/mradaideh/neorl

	.. image:: ../images/fork_button_red.png
	   :scale: 70 %
	   :alt: alternate text
	   :align: center

#. Once this is done, a new repository will be created under your ownership. It will exist under `github_username/neorl`. From here, the forked repository can be cloned like any other repository. Navigate to a directory you plan on working in and enter the command: ``git clone git@github.com:github-un/neorl.git``
   
   Replace the **github-un** with your Github username

#. From here, the forked repository can be committed to using typical `git` practices.

Incorporating Changes From the Main Repository
-------------------------------------------------

Often when making large contributions, it may be necessary to work on the fork for an extended period of time where updates have been pushed to the upstream (`mradaideh/neorl`) repository after you originally created the fork. In order to incorporate these new updates into your local fork:

#. Commit and push all work in your local repository with regular ``git commit`` and ``git push`` commands.
#. First add the upstream repository into known remote repositories:
   ``git remote add upstream https://github.com/mradaideh/neorl``
#. Fetch changes that have been made to upstream repository:
   ``git fetch upstream``
#. Merge changes made to upstream repository into your local repository:
   ``git merge upstream/master master``
#. Push from your local fork (on your PC) to your remote fork (on Github):
   ``git push``

Submitting a Pull Request
---------------------------
Once a major contribution to the NEORL code base has been developed and is ready to be incorporated into 
the upstream repository (`mradaideh/neorl`), a pull request can be submitted:

#. While logged into the account with the forked repository containing the changes, navigate to the Github 
   page for the upstream NEORL repository: https://github.com/mradaideh/neorl
   
#. Select the "Pull requests" tab near the top of the page and select "new pull request".

	.. image:: ../images/annotate_prerequest.png
	   :scale: 70 %
	   :alt: alternate text
	   :align: center

#. Select "compare across forks" link.

	.. image:: ../images/compare_across.png
	   :scale: 70 %
	   :alt: alternate text
	   :align: center
   
#. From here, the "base repository" should be set to `mradaideh/neorl/master` and the "head repository"
   should point to the repository that is under your name.
   
#. Press the "Create pull request" button and fill out the submission fields and submit it for review!




