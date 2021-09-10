.. _random:

.. automodule:: neorl.tune.randtune

Random Search
===============

A module for random search of hyperparameters of NEORL algorithms. 

Original paper: Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

Random search is a technique where random combinations of the hyperparameters are used to find the best solution for the algorithm used. Random search tries random combinations of the hyperparameters, where the cost function is evaluated at these random sets in the parameter space. As indicated by the reference above, the chances of finding the optimal hyperparameters are comparatively higher in random search than grid search. This is because of the random search pattern, as the algorithm might end up being used on the optimized hyperparameters without any aliasing or wasting of resources.

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: RANDTUNE
  :members:
  :inherited-members:
  
Example
-------

Example of using random search to tune three ES hyperparameters for solving the 5-d Sphere function 

.. literalinclude :: ../scripts/ex_random.py
   :language: python

Notes
-----

- For ``ncores > 1``, the parallel tuning engine starts. **Make sure to run your python script from the terminal NOT from an IDE (e.g. Spyder, Jupyter Notebook)**. IDEs are not robust when running parallel problems with packages like ``joblib`` or ``multiprocessing``. For ``ncores = 1``, IDEs seem to work fine.    
- Random search struggles with dimensionality if there are large number of hyperparameters to tune. Therefore, it is always recommended to do a preliminary sensitivity study to exclude or fix the hyperparameters with small impact.      
- To determine an optimal ``ncases``, try to setup your problem for grid search on paper, calculate the grid search ``ncases``, and go for 50\% of this number. Achieving similar performance with 50\% cost is a promise for random search.  
- For difficult problems, the analyst can start with a random search first to narrow the choices of the important hyperparameters. Then, a grid search can be executed on those important parameters with more refined and narrower grids. 


 
