.. _evolsearch:

.. automodule:: neorl.tune.estune

Evolutionary Search
=====================

A module of evolutionary search for hyperparameter tuning of NEORL algorithms.

Original paper: E. Bochinski, T. Senst and T. Sikora, "Hyper-parameter optimization for convolutional neural network committees based on evolutionary algorithms," 2017 IEEE International Conference on Image Processing (ICIP), Beijing, China, 2017, pp. 3924-3928, doi: 10.1109/ICIP.2017.8297018.

We have used a compact evolution strategies (ES) module for the purpose of tuning the hyperparameters of NEORL algorithms. See the :ref:`ES algorithm <es>` section for more details about the (:math:`\mu,\lambda`) algorithm. ES tuner leverages a population of individuals, where each individual represents a sample from the hyperparameter space. ES uses recombination, crossover, and mutation operations to improve the individuals from generation to the other. The best of the best individuals in all generations are reported as the top hyperparameter sets to use further with the algorithm. 


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: ESTUNE
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_evolu.py
   :language: python

Notes
-----
- Evolutionary search uses fixed values for ``lambda_=10`` and ``mu=10``. 
- Therefore, the total cost of evolutionary search or the total number of hyperparameter tests is ``ngen * 10``.
- For categorical variables, use integers to encode them as integer variables. Then, inside the ``tune_fit`` function, the integers are converted back to the real categorical value. See how ``speed_mech`` is handled in the example above.  
- The strategy and individual vectors in the ES tuner are updated similarly to the ES algorithm module described :ref:`here <es>`.    
- For difficult problems, the analyst can start with a random search first to narrow the choices of the important hyperparameters. Then, an evolutionary search can be executed on those important parameters to refine their values. 



 
