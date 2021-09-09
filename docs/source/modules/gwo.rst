.. _gwo:

.. automodule:: neorl.evolu.gwo


Grey Wolf Optimizer (GWO)
===============================================

A module for the Grey Wolf Optimizer with parallel computing support. 

Original paper: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey wolf optimizer. Advances in engineering software, 69, 46-61.

.. image:: ../images/gwo.jpg
   :scale: 40%
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

.. autoclass:: GWO
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_gwo.py
   :language: python

Notes
-----

- GWO assigns the best fitness to the first wolf (called **Alpha**), second best fitness to **Beta** wolf, third best fitness to **Delta** wolf, while the remaining wolves in the group are called **Omega**, which follow the leadership and position of Alpha, Beta, and Delta.  
- ``ncores`` argument evaluates the fitness of all wolves in the group in parallel. Therefore, set ``ncores <= nwolves`` for most optimal resource allocation.
- Look for an optimal balance between ``nwolves`` and ``ngen``, it is recommended to minimize the number of ``nwolves`` to allow for more updates and more generations.
- Total number of cost evaluations for GWO is ``nwolves`` * ``ngen``.