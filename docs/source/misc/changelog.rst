.. _changelog:

Changelog
==========

Release 1.2.0 (2021-03-15)
--------------------------

- Added DE with serial implementation.
- Added NES with parallel implementation. 
- Restructuring the input parameter space.
- Detailed README file in the Github page. 
- Added unit tests to NEORL.

Release 1.1.0 (2020-12-15)
--------------------------

- Added Genetic/Evolutionary hyperparameter tuning.
- Added Bayesian hyperparameter tuning from ``scikit-optimise``.
- Added parallel evolutionary strategies(ES).
- Added hybrid PPO-ES algorithm.
- Updated documentation. 

Release 1.0.0 (2020-09-15)
--------------------------

- Added parallel GA.
- Added a PDF documentation. 
- Added parallel PSO.
- Added Random search hyperparameter tuning.


Release 0.1.1 (2020-03-15)
--------------------------

- A support for both classical (evolutionary) and modern (machine learning) optimization in the same package. Currently, DQN (serial), PPO (parallel), A2C (parallel), GA (serial), SA (serial) are supported. All RL algorithms are based upon ``stable-baselines``.
-  Easy-to-use syntax and friendly interaction with the package.
-  A support for parallel computing. 
-  Efficient on-the-fly progress monitoring and detailed output postprocessing for handy usage.
-  Added grid search hyperparameter tuning.
-  For developers: an organized implementation and source code structure to facilitate the job of future external contributors.
-  NEORL Examples are provided in the example directory.