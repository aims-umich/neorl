.. _install:

Quick Installation
======================

Use this guide if you are an expert Python user and aware of Python virtual environment and package management. For a safe and clean installation guide, see the :ref:`Detailed Installation <detinstall>` section. 

Prerequisites
--------------

NEORL is tested on ``python3 (3.6-3.7)`` with the development headers. **Please, avoid using python 3.5 or lower** (as dictionary ordering is not preserved), or **python 3.8 or newer** (as tensorflow-1.14.0 will not be stable).

.. note::

    NEORL supports ``tensorflow`` versions from 1.8.0 to 1.14.0, **we do not support tensorflow >= 2.0**. Please, make sure to uninstall ``tensorflow`` if already installed on your environment or have a proper version. If ``tensorflow`` is left in the virtual environment, NEORL will automatically force ``tensorflow-1.14.0`` for most stability.

Ubuntu Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  sudo apt-get update && sudo apt-get install cmake python3-dev

Windows 10 Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

To install NEORL on Windows, it is recommended to install Anaconda3 on the machine first to have some pre-installed packages, then open "Anaconda Prompt" as an administrator and use the instructions below for **Install using pip**.

.. note::

	You can access Anaconda3 archives for all OS installers from this page https://repo.anaconda.com/archive/

.. note::

	We typically recommend creating a new virtual environment for NEORL to avoid version conflicts and compatibility issues with other projects.
	
	.. code-block:: bash
	
		conda create --name neorl python=3.7
		conda activate neorl

Install using pip
--------------------

For both Ubuntu and Windows, you can install NEORL via pip

.. code-block:: bash
	
    pip install neorl