.. _changelog:

Change Log
==============

Stable Releases (for users)
------------------------------

Release 1.7 (2021-11-24)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.7

- Fixed a major bug following scikit-learn update to 1.0.0. Now, NEORL supports scikit-learn <= 0.24.0
- Misc. minor updates for the documentation and the source code 

Release 1.6 (2021-09-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.6

- The first NEORL stable release. 
- Includes all changes in all previous beta releases: 1.2.0b-1.5.7b.
- Summary: 28 algorithms, 4 tuning methods, 10 real-world examples, and 39 unit tests. 
- Documentation and Github repo are up-to-date. 

Beta Releases (for developers)
---------------------------------

Release 1.7.1b (2022-1-03)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.7.1b --extra-index-url https://test.pypi.org/simple

- Added new NEORL example 11 on nuclear microreactor application. 
- Documentation structure updates. Now subsections are part of the documentation structure. 

Release 1.6.2b (2021-10-07)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.6.2b --extra-index-url https://test.pypi.org/simple

- Removed summary files from RL runners.
- Added a capability to save current model for RL runners. Currently best model and last model are saved. 

Release 1.6.1b (2021-09-20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.6.1b --extra-index-url https://test.pypi.org/simple

- Fixed a bounding check bug in FNEAT and RNEAT.
- Fixed different typos in the documentation. 
- Increased the width of the online documentation page to fit more code/words. 

Releases 1.5.3b-1.5.7b (2021-09-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.5.7b --extra-index-url https://test.pypi.org/simple

- Added hybrid neuroevolution algorithm: Neural genetic algorithm (NGA)
- Added hybrid neuroevolution algorithm: Neural Harris hawks optimization (NHHO)
- Added Cuckoo Search with all spaces handled.
- Added Ant Colony optimization for continuous domains.
- Added Tabu Search for discrete domains.
- Fixed a critical bug in the terminal API in the followup 1.5.4b
- Fixed a bug in the terminal API continue mode in the followups 1.5.5b-1.5.6b.
- Fixed hyperthreading issue for RL algorithms in the followup 1.5.7b.

Release 1.5.2b (2021-08-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.5.2b --extra-index-url https://test.pypi.org/simple

- Added hybrid neuroevolution algorithm PPO-ES.
- Added hybrid neuroevolution algorithm ACKTR-DE.
- Updated documentation for RL algorithms.

Release 1.5.1b (2021-08-01)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.5.1b --extra-index-url https://test.pypi.org/simple

- Added RNEAT and FNEAT with full documentation.
- Added mixed discrete optimization to WOA, GWO, SSA, DE, MFO, JAYA, PESA2
- Added friendly implementation to construct parallel environments for RL: DQN, ACKTR, A2C, PPO

Release 1.5.0b (2021-07-28)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.5.0b --extra-index-url https://test.pypi.org/simple

- Updated Example 1 on using RL to solve Travel Salesman problem
- Added Example 10 on using RL to solve Knapsack problem
- Added CEC-2008 benchmark functions for large-scale optimization

Release 1.4.8b (2021-07-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.4.8b --extra-index-url https://test.pypi.org/simple

- Added environment class constructor for DQN, ACER, PPO, ACKTR, A2C
- Added mixed discrete/continuous optimization for PPO, ACKTR, A2C
- Added categorical/discrete optimization for ACER, DQN.

Releases 1.4.6b-1.4.7b (2021-07-09)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.4.7b --extra-index-url https://test.pypi.org/simple

- Modifying Bat algorithm to handle mixed spaces. 
- Added Example 6 on three-bar truss design.
- Added Examples 7 and 8 on pressure vessel design. 
- Added Example 9 on cantilever stepped beam.
- Fixing bugs after 1.4.6b.

Releases 1.4.1b-1.4.5b (2021-07-05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  pip install neorl==1.4.5b --extra-index-url https://test.pypi.org/simple
  
- Fixing bounding issues in most evolutionary algorithms.
- Fixing PESA/PESA2 parallel mode.
- Replacing XNES with WOA in modern PESA2.
- Added a module for Harris Hawks Optimization.
- Added the BAT algorithm.
- Removed deprecation warnings of Tensorflow from NEORL.
- Added a module for JAYA.
- Added a module for MFO.

Old Releases (outdated)
------------------------

Release 1.4.0b (2021-05-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added a module for Simulated Annealing (SA).
- Added a Genetic/Evolutionary hyperparameter tuning module.
- Added ACER module for RL optimization.
- Added ACKTR module for RL optimization.
- Added a WOA module for evolutionary optimization. 
- Added a SSA module for evolutionary optimization. 

Release 1.3.5b (2021-05-10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added CEC'2017 Test Suite benchmarks
- Added a set of classical mathematical functions
- Added new Example (4) on the website on how to access the benchmarks
- Added new Example (5) on the website on how to optimize the benchmarks

Releases 1.3.1b-1.3.2b (2021-05-4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Fixing miscellaneous bugs

Release 1.3.0b (2021-05-1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added a module for the hybrid algorithm PESA.
- Added a module for the modern hybrid algorithm PESA2.
- Added a GWO module. 
- Adding min/max modes for all algorithms.

Release 1.2.0b (2021-04-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **The first public open-source version of NEORL**
- Added DE with serial implementation.
- Added XNES with parallel implementation. 
- Restructuring the input parameter space.
- Detailed README file in the Github page. 
- Added unit tests to NEORL.
- Automatic documentation via Sphinx

Release 1.1.0-Private (2020-12-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added Bayesian hyperparameter tuning from ``scikit-optimise``.
- Added parallel evolutionary strategies(ES).
- Updated documentation. 

Release 1.0.0-Private (2020-09-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Added evolutionary strategies ES.
- Added a local PDF documentation. 
- Added parallel PSO.
- Added Random search hyperparameter tuning.

Release 0.1.1-Private (2020-03-15)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- A support for both classical (evolutionary) and modern (machine learning) optimization in the same package. Currently, DQN (serial), PPO (parallel), A2C (parallel), GA (serial), SA (serial) are supported. All RL algorithms are based upon ``stable-baselines``.
-  Easy-to-use syntax and friendly interaction with the package.
-  A support for parallel computing. 
-  Added grid search hyperparameter tuning.
-  For developers: an organized implementation and source code structure to facilitate the job of future external contributors.
-  NEORL examples are provided in the "examples" directory.