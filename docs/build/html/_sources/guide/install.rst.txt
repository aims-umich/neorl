.. _install:

Quick Installation
======================

Use this guide if you are an expert Python user and aware of Python virtual environment and package management. For a safe and clean installation guide, see the :ref:`Detailed Installation <detinstall>` section. 

Prerequisites
--------------

NEORL is tested on ``python3 (3.10-3.13)`` with the development headers. **Please avoid using newer Python versions**, as ``tensorflow-2.21.0`` will be unstable. For older Python versions, please use **NEORL 1.8**.

.. note::

    NEORL supports ``tensorflow`` versions from ``2.21.0``. Please make sure to uninstall ``tensorflow`` if it is already installed in your environment, or ensure you have a compatible version. If ``tensorflow`` is left in the virtual environment, NEORL will automatically force ``tensorflow-2.21.0`` for maximum stability. If you require older ``tensorflow`` versions, please use **NEORL 1.8**.

Ubuntu Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  sudo apt-get update && sudo apt-get install cmake python3-dev

Windows 10-11 Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install NEORL on Windows, it is recommended to install Anaconda3 on the machine first to have some pre-installed packages, then open "Anaconda Prompt" as an administrator and use the instructions below for **Install using pip**.

.. note::

	You can access Anaconda3 archives for all OS installers from this page https://repo.anaconda.com/archive/

.. note::

	We typically recommend creating a new virtual environment for NEORL to avoid version conflicts and compatibility issues with other projects.
	
	.. code-block:: bash
	
		conda create --name neorl python=3.13
		conda activate neorl

Install using pip
--------------------

For both Ubuntu and Windows, you can install NEORL via pip

.. code-block:: bash
	
    pip install neorl