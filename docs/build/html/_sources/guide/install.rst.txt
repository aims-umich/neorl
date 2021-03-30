.. _install:

Installation
============

Prerequisites
--------------

NEORL requires python3 (>=3.5) with the development headers.

.. note::

    NEORL supports Tensorflow versions from 1.8.0 to 1.14.0 (Recommended). Please, make sure to have the proper TensorFlow installed on your machine before installing NEORL. You can install a specific Tensorflow version via pip using

	.. code-block:: bash
	
	    pip install tensorflow==1.14.0	

Ubuntu Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  sudo apt-get update && sudo apt-get install cmake python3-dev

Mac OS X Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

NEORL is not tested yet on Mac OS, the Mac OS NEORL is currently under development. 


Windows 10 Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

To install NEORL on Windows, it is recommended to install Anaconda3 on the machine first to have some pre-installed packages, then open "Anaconda Prompt" as an administrator and use the instructions below for "Install using pip".

.. note::

	You can access Anaconda3 archives for all OS installers from this page https://repo.anaconda.com/archive/


Install using pip
--------------------

For both Ubuntu and Windows, you can install NEORL via pip

.. code-block:: bash

    pip install neorl