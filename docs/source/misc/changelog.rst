.. _changelog:

Changelog
==========

Release 1.5.3b (correction: 1.5.4b) (2021-08-22)
--------------------------------------------------

- Added hybrid neuroevolution algorithm: Neural genetic algorithm (NGA)
- Added hybrid neuroevolution algorithm: Neural Harris hawks optimization (NHHO)
- Added Cuckoo Search with all spaces handled.
- Added Ant Colony optimization for continuous domains.
- Added Tabu Search for discrete domains.
- Fixed a critical bug in the terminal API in the followup 1.5.4b

Release 1.5.2b (2021-08-10)
------------------------------------

- Added hybrid neuroevolution algorithm PPO-ES.
- Added hybrid neuroevolution algorithm ACKTR-DE.
- Updated documentation for RL algorithms.

Release 1.5.1b (2021-08-01)
------------------------------------

- Added RNEAT and FNEAT with full documentation.
- Added mixed discrete optimization to WOA, GWO, SSA, DE, MFO, JAYA, PESA2
- Added friendly implementation to construct parallel environments for RL: DQN, ACKTR, A2C, PPO

Release 1.5.0b (2021-07-28)
------------------------------------

- Updated Example 1 on using RL to solve Travel Salesman problem
- Added Example 10 on using RL to solve Knapsack problem
- Added CEC-2008 benchmark functions for large-scale optimization

Release 1.4.8b (2021-07-14)
------------------------------------

- Added environment class constructor for DQN, ACER, PPO, ACKTR, A2C
- Added mixed discrete/continuous optimization for PPO, ACKTR, A2C
- Added categorical/discrete optimization for ACER, DQN.

Release 1.4.6b-1.4.7b (2021-07-09)
------------------------------------

- Modifying Bat algorithm to handle mixed spaces. 
- Added Example 6 on three-bar truss design.
- Added Examples 7 and 8 on pressure vessel design. 
- Added Example 9 on cantilever stepped beam.
- Fixing bugs after 1.4.6b.

Release 1.4.5b (2021-07-05)
------------------------------------

- Fixing bounding issues in most evolutionary algorithms.

Release 1.4.4b (2021-06-30)
------------------------------------

- Fixing PESA/PESA2 parallel mode.
- Replacing XNES with WOA in modern PESA2.
- Added a module for Harris Hawks Optimization.

Release 1.4.3b (2021-06-24)
------------------------------------

- Added the BAT algorithm.

Release 1.4.2b (2021-06-17)
------------------------------------

- Removed deprecation warnings of Tensorflow from NEORL.

Release 1.4.1b (2021-06-15)
------------------------------------

- Added a module for JAYA.
- Added a module for MFO.

Release 1.4.0b (2021-05-15)
------------------------------------

- Added a module for Simulated Annealing (SA).
- Added a Genetic/Evolutionary hyperparameter tuning module.
- Added ACER module for RL optimization.
- Added ACKTR module for RL optimization.
- Added a WOA module for evolutionary optimization. 
- Added a SSA module for evolutionary optimization. 

Release 1.3.5b (2021-05-10)
------------------------------------

- Added CEC'2017 Test Suite benchmarks
- Added a set of classical mathematical functions
- Added new Example (4) on the website on how to access the benchmarks
- Added new Example (5) on the website on how to optimize the benchmarks

Release 1.3.1b/1.3.2b (2021-05-4)
------------------------------------

- Fixing miscellaneous bugs

Release 1.3.0b (2021-05-1)
---------------------------

- Added a module for the hybrid algorithm PESA.
- Added a module for the modern hybrid algorithm PESA2.
- Added a GWO module. 
- Adding min/max modes for all algorithms.

Release 1.2.0b (2021-04-15)
---------------------------

- **The first public open-source version of NEORL**
- Added DE with serial implementation.
- Added XNES with parallel implementation. 
- Restructuring the input parameter space.
- Detailed README file in the Github page. 
- Added unit tests to NEORL.
- Automatic documentation via Sphinx

Release 1.1.0-Private (2020-12-15)
------------------------------------

- Added Bayesian hyperparameter tuning from ``scikit-optimise``.
- Added parallel evolutionary strategies(ES).
- Updated documentation. 

Release 1.0.0-Private (2020-09-15)
-----------------------------------

- Added evolutionary strategies ES.
- Added a local PDF documentation. 
- Added parallel PSO.
- Added Random search hyperparameter tuning.

Release 0.1.1-Private (2020-03-15)
-----------------------------------

- A support for both classical (evolutionary) and modern (machine learning) optimization in the same package. Currently, DQN (serial), PPO (parallel), A2C (parallel), GA (serial), SA (serial) are supported. All RL algorithms are based upon ``stable-baselines``.
-  Easy-to-use syntax and friendly interaction with the package.
-  A support for parallel computing. 
-  Added grid search hyperparameter tuning.
-  For developers: an organized implementation and source code structure to facilitate the job of future external contributors.
-  NEORL examples are provided in the "examples" directory.