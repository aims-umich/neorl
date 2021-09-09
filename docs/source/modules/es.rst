.. _es:

.. automodule:: neorl.evolu.es


Evolution Strategies (:math:`\mu,\lambda`) (ES)
================================================

A module for the evolution strategies (:math:`\mu,\lambda`) with adaptive strategy vectors. 

Original paper: Bäck, T., Fogel, D. B., Michalewicz, Z. (Eds.). (2018). Evolutionary computation 1: Basic algorithms and operators. CRC press.

.. image:: ../images/es.png
   :scale: 35%
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

.. autoclass:: ES
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_es.py
   :language: python

Notes
-----

- Too large population mutation rate ``mutpb`` could destroy the population, the recommended range for this variable is between 0.01-0.4. 
- Too large ``smax`` will allow the individual to be perturbed in a large rate.  
- Too small ``cxpb`` and ``mutpb`` reduce ES exploration, and increase the likelihood of falling in a local optima.
- Usually, population size ``lambda_`` between 60-100 shows good performance along with ``mu=0.5*lambda_``. 
- Look for an optimal balance between ``lambda_`` and ``ngen``, it is recommended to minimize population size to allow for more generations.
- Total number of cost evaluations for ES is ``npop`` * ``ngen``.
- ``cxmode='blend'`` with ``alpha=0.5`` may perform better than ``cxmode='cx2point'``.