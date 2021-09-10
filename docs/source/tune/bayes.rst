.. _bayes:

.. automodule:: neorl.tune.bayestune

Bayesian Search
=========================

A module of Bayesian optimisation search for hyperparameter tuning of NEORL algorithms based upon ``scikit-optimise``. 

Original paper: https://arxiv.org/abs/1012.2599

Bayesian search, in contrast to grid and random searches, keeps track of past evaluation results. Bayesian uses past evaluations to form a probabilistic model mapping hyperparameters to a probability of a score on the objective function (e.g. max/min fitness). Bayesian optimization excels when the objective functions are expensive to evaluate, when we do not have access to derivatives, or when the problem at hand is non-convex. 

What can you use?
--------------------

-  Multi processing: ✔️ (Multithreading in a single processor)
-  Discrete/Continuous/Mixed spaces: ✔️
-  Reinforcement Learning Algorithms: ✔️
-  Evolutionary Algorithms: ✔️
-  Hybrid Neuroevolution Algorithms: ✔️

Parameters
----------

.. autoclass:: BAYESTUNE
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_bayes.py
   :language: python

Notes
-----

- We allow a weak parallelization of Bayesian search via multithreading. The user can start independent Bayesian search with different seeds by increasing ``ncores``. However, all threads will be executed on a single processor, which will slow down every Bayesian sequence. Therefore, this option is recommended when each hyperparameter case is fast-to-evaluate and does not require intensive CPU power. 
- If the user sets ``ncores=4`` and sets ``ncases=15``, a total of 60 hyperparameter cases are evaluated, where each thread uses 25\% of the CPU power. **The extension to multiprocessing/multi-core capability is on track in future**.     
- Keep ``ncases >= 11``. If ncases < 11, the optimiser resets ``ncases=11``. It is good to start with ``ncases=30``, check the optimizer convergence, and increase as needed.
- Relying on ``grid/categorical`` variables can accelerate the search by a wide margin. Therefore, if the user is aware of certain values of the (``int/discrete``) or the (``float/continuous``) hyperparameters, it is good to convert them to ``grid/categorical``.

Acknowledgment
-----------------

Thanks to our fellows in `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`_, as we used their ``gp_minimize`` implementation to leverage our Bayesian search module in our framework.

Head, Tim, Gilles Louppe MechCoder, and Iaroslav Shcherbatyi. "scikit-optimize/scikit-optimize: v0.7.1"(2020).


