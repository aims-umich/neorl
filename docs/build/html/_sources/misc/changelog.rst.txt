.. _changelog:

Changelog
==========

Coming next
--------------------------

- To add a module for the hybrid algorithm PESA.
- To add Genetic/Evolutionary hyperparameter tuning with documentation.
- To add a module for the hybrid PPO-ES algorithm.
- Adding a class for easy-to-use environment construction for RL optimization.
- And more ...

Release 1.2.0 (2021-04-15)
---------------------------

- **The first public open-source version of NEORL**
- Added DE with serial implementation.
- Added NES with parallel implementation. 
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

- Added parallel GA.
- Added a PDF documentation. 
- Added parallel PSO.
- Added Random search hyperparameter tuning.


Release 0.1.1-Private (2020-03-15)
-----------------------------------

- A support for both classical (evolutionary) and modern (machine learning) optimization in the same package. Currently, DQN (serial), PPO (parallel), A2C (parallel), GA (serial), SA (serial) are supported. All RL algorithms are based upon ``stable-baselines``.
-  Easy-to-use syntax and friendly interaction with the package.
-  A support for parallel computing. 
-  Efficient on-the-fly progress monitoring and detailed output postprocessing for handy usage.
-  Added grid search hyperparameter tuning.
-  For developers: an organized implementation and source code structure to facilitate the job of future external contributors.
-  NEORL Examples are provided in the example directory.