.. _grid:

.. automodule:: neorl.tune.gridtune

Grid Search
=============

A module for grid search of hyperparameters of NEORL algorithms. 

Original paper: Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

Grid Search is an exhaustive search for selecting an optimal set of algorithm hyperparameters. In Grid Search, the analyst sets up a grid of hyperparameter values. A multi-dimensional full grid of all hyperparameters is constructed, which contains all possible combinations of hyperparameters. Afterwards, every combination of hyperparameter values is tested in serial/parallel, where the optimization score (e.g. fitness) is estimated. Grid search can be very expensive for fine grids as well as large number of hyperparameters to tune. 

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: GRIDTUNE
  :members:
  :inherited-members:
  
Example
-------

Example of using grid search to tune three ES hyperparameters for solving the 5-d Sphere function 

.. literalinclude :: ../scripts/ex_grid.py
   :language: python

Notes
-----

- For ``ncores > 1``, the parallel tuning engine starts. **Make sure to run your python script from the terminal NOT from an IDE (e.g. Spyder, Jupyter Notebook)**. IDEs are not robust when running parallel problems with packages like ``joblib`` or ``multiprocessing``. For ``ncores = 1``, IDEs seem to work fine.    
- If there are large number of hyperparameters to tune (large :math:`d`), try nested grid search. First, run a grid search on few parameters first, then fix them to their best, and start another grid search for the next group of hyperparameters, and so on.    
- Always start with coarse grid for all hyperparameters (small :math:`k_i`) to obtain an impression about their sensitivity. Then, refine the grids for those hyperparameters with more impact, and execute a more detailed grid search.  
- Grid search is ideal to use when the analyst has prior experience on the feasible range of each hyperparameter and the most important hyperparameters to tune. 
