.. _woa:

.. automodule:: neorl.evolu.woa


Whale Optimization Algorithm (WOA)
===============================================

A module for the Whale Optimization Algorithm with parallel computing support. 

Original paper: Mirjalili, S., Lewis, A. (2016). The whale optimization algorithm. Advances in engineering software, 95, 51-67.

.. image:: ../images/woa.jpg
   :scale: 60%
   :alt: alternate text
   :align: center

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: WOA
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_woa.py
   :language: python

Notes
-----

- WOA mimics the social behavior of humpback whales, which is inspired by the bubble-net hunting strategy.  
- The whale leader is controlled by multiple coefficients, where ``a`` is considered the most important. The coefficient ``a`` balances WOA exploration and exploitation. The value of ``a`` is annealed "linearly" from ``a0 > 0`` to 0 over the course of ``ngen``. Typical values for ``a0`` are 1, 1.5, 2, and 4.
- Therefore, the user should notice that ``ngen`` value used within the ``.evolute`` function has an impact on the ``a`` value and hence on WOA overall performance.
- ``ncores`` argument evaluates the fitness of all whales in the population in parallel. Therefore, set ``ncores <= nwhales`` for most optimal resource allocation.
- Look for an optimal balance between ``nwhales`` and ``ngen``, it is recommended to minimize the number of ``nwhales`` to allow for more updates and more generations.
- Total number of cost evaluations for WOA is ``nwhales`` * ``(ngen + 1)``.