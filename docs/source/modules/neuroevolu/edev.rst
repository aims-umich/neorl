.. _edev:

.. automodule:: neorl.hybrid.edev

Ensemble of Differential Evolution Variants (EDEV)
=======================================================

A powerful hybrid ensemble of three differential evolution variants: JADE (adaptive differential evolution with optional external archive), CoDE (differential evolution with composite trial vector generation strategies and control parameters), and EPSDE (differential evolution algorithm with ensemble of parameters and mutation strategies).


Original paper: Wu, G., Shen, X., Li, H., Chen, H., Lin, A., Suganthan, P. N. (2018). Ensemble of differential evolution variants. Information Sciences, 423, 172-186.


What can you use?
--------------------

- Multi processing: ✔️
- Discrete spaces: ✔️
- Continuous spaces: ✔️
- Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: EDEV
  :members:
  :inherited-members:

Example
-------

.. literalinclude :: ../../scripts/ex_edev.py
   :language: python

Notes
-----

- Choosing the value of ``npop`` and ``lambda_`` should be careful for EDEV. The third variant (EPSDE) performs mutation and crossover on 6 different individuals. Therefore, make sure that the second and third sub-populations have more than 6 individuals for optimal performance, or simply ensure that ``int(npop * lambda_) > 6``.

- Increasing ``lambda_`` value will make the size of the three sub-populations comparable. In the original paper, one population is bigger than the others, so for ``lambda_ = 0.1`` and ``npop=100``, the three sub-populations have sizes 80, 10, and 10.  

- The parallelization of EDEV is bottlenecked by the size of the sub-populations. For example, for sub-populations of size 80, 10, and 10, using ``ncores = 80`` will ensure that the first sub-population is executed in one round, but the other two sub-populations will be evaluated in sequence with 10 cores only.  

- Unlike standalone DE, for EDEV, the values of the hyperparameters ``F`` and ``CR`` are automatically adapted. 
    
- The parameter ``ng`` in the ``evolute`` function helps to determine the frequency/period at which the three sub-populations are swapped between the DE variants based on their prior performance. For example, if ``ngen = 100`` and ``ng=20``, five updates will occur during the full evolution process. 

- Look for an optimal balance between ``npop`` and ``ngen``, it is recommended to keep population size optimal to allow for more generations, but also sufficient to keep the three sub-populations active.

- As EDEV is a bit complex, the total number of cost evaluations for EDEV can be accessed via the returned dictionary key: ``edev_hist['F-Evals']`` in the example above.
