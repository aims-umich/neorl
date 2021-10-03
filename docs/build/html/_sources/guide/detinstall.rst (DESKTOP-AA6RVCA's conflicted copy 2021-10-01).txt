.. _detinstall:

Detailed Installation
======================

Use this guide if you are looking for a safe and clean installation of NEORL with all Python tools and package management. If you are an expert Python user and aware of Python virtual environment and package management, see the :ref:`Quick Installation <install>` section.  

Linux/Ubuntu
------------------

Step 0: Prerequisites (Anaconda3 Installation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda3 will provide you with OS-independent framework that hosts Python packages, including NEORL. **If you have Anaconda3 installed on your machine, move to Step 1**. 

1- First download the Anaconda3 package

.. code-block:: bash
	
	wget --no-check-certificate https://repo.continuum.io/archive/Anaconda3-2019.03-Linux-x86_64.sh
	
2- Start the installer

.. code-block:: bash
	
	bash Anaconda3-2019.03-Linux-x86_64.sh
	
3- Follow the instructions on the screen and wait until completion (See the notes below on how to respond to certain prompts)

.. note::

	Choose the default location for installation when asked (e.g. /home/username/anaconda3)

.. note::

	Enter **yes** when you get this prompt: **Do you wish the installer to initialize Anaconda3 by running conda init? [yes|no]**	


4- You may update the setup tool packages before proceeding

.. code-block:: bash
	
	pip install --upgrade pip setuptools wheel
	
	    
Step 1: Create virtual environment for NEORL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NEORL is tested on ``python3 (3.5-3.7)`` with the development headers.

1- Create a new python-3.7 environment with name ``neorl``

.. code-block:: bash
	
	conda create --name neorl python=3.7	

.. warning::

	For some machines that are not updated frequently (e.g. clusters), TensforFlow may fail to load due to outdated gcc libraries. If you encounter those errors, we typically recommend to downgrade python by using python=3.6 or python=3.5 when creating the virtual environment.  
	
2- Activate the environment 

.. code-block:: bash
	
	conda activate neorl
	
.. warning::

	You need to run ``conda activate neorl`` every time you log in the system, therefore, it is good to add this command to your OS bashrc or environment variables for automatic activation when you log in.

Step 2: Install NEORL
~~~~~~~~~~~~~~~~~~~~~~~~

Make sure ``neorl`` environment is activated, then run the following command:
 
.. code-block:: bash

  pip install neorl
	
.. warning::

	Depending on your OS, ``conda`` command may fail due to unknown reasons. If ``conda list`` command fails, then type
	
	.. code-block:: bash
	
		conda update -n base -c defaults conda  

Step 3: Test NEORL
~~~~~~~~~~~~~~~~~~~~~~~~

After an error-free Step 2 completion, you can test NEORL by typing on the terminal:

.. code-block:: bash

  neorl
  
which yields NEORL logo

.. image:: ../images/neorl_terminal.jpg
   :scale: 35 %
   :alt: alternate text
   :align: center

and you can run unit tests by running:

.. code-block:: bash

  neorl --test
  

Windows 10
------------------

Step 0: Prerequisites (Anaconda3 Installation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Anaconda3 will provide you with OS-independent framework that hosts Python packages, including NEORL. **If you have Anaconda3 installed on your machine, move to Step 1**. 

1- First download the Anaconda3 package by visiting the link in the note below and search for ``Anaconda3-2019.03-Windows-x86_64.exe``

.. note::

	You can access Anaconda3 archives for all OS installers from this page https://repo.anaconda.com/archive/
	
or simply click on the link below to download:

https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe
	
2- Start the exe installer, follow the instructions on the screen, and wait until completion. See the notes below on what options to choose. 

.. note::

	- Choose the option "Register Anaconda as your default Python-3.7". 
	- For the option of "adding anaconda to your PATH variables", choose this option only if you have cleaned all previous Anaconda3 releases from your machine. 	
	
	    
Step 1: Create virtual environment for NEORL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Search for ``Anaconda Prompt`` and open a new terminal as an administrator  

1- Create a new python-3.7 environment with name ``neorl``

.. code-block:: bash
	
	conda create --name neorl python=3.7	
	
2- Activate the environment 

.. code-block:: bash
	
	conda activate neorl

Step 2: Install NEORL
~~~~~~~~~~~~~~~~~~~~~~~~

Make sure ``neorl`` environment is activated, then run the following command:
 
.. code-block:: bash

  pip install neorl

.. warning::

	Depending on your OS, ``conda`` command may fail due to unknown reasons. If ``conda list`` command fails, then type
	
	.. code-block:: bash
	
		conda update -n base -c defaults conda

Step 3: Test NEORL
~~~~~~~~~~~~~~~~~~~~~~~~

After an error-free Step 2 completion, you can test NEORL by typing on the terminal:

.. code-block:: bash

  neorl
  
which yields NEORL logo

.. image:: ../images/neorl_terminal.jpg
   :scale: 35 %
   :alt: alternate text
   :align: center

and you can run unit tests by running:

.. code-block:: bash

  neorl --test

.. warning::

	You need to run ``conda activate neorl`` every time you log in the system, therefore, it is good to add this command to your OS environment variables for automatic activation. Similarly, make sure to connect your Jupyter notebook and Spyder IDE to ``neorl`` virtual environment NOT to the default ``base``.