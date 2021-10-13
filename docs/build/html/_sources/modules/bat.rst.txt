.. _bat:

.. automodule:: neorl.evolu.bat


Bat Algorithm (BAT)
===============================================

A module for the bat optimization algorithm with differential operator, Levy flights trajectory, and parallel computing support. 

Original papers: 

- Xie, J., Zhou, Y., Chen, H. (2013). A novel bat algorithm based on differential operator and Lévy flights trajectory. Computational intelligence and neuroscience, 2013.

- Yang, X. S. (2010). A new metaheuristic bat-inspired algorithm. In Nature inspired cooperative strategies for optimization (NICSO 2010) (pp. 65-74). Springer, Berlin, Heidelberg.

.. image:: ../images/bat.png
   :scale: 50%
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

.. autoclass:: BAT
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../scripts/ex_bat.py
   :language: python

Notes
-----

- BAT mimics the echolocation behavior of bats in nature. The bats emit a very loud and short sound pulse; the echo that reflects back from the surrounding objects is received by their big ears. This feedback information of echo is analyzed by the bats, which reveals the direction of the flight pathway. The echo also helps the bats to distinguish different insects and obstacles to hunt prey, where the search for the prey here is analogous to the search for global optima.
- The bats start with loudness value of ``A``, and decay it by a factor of ``alpha``. If the user chooses ``alpha=1``, fixed value of ``A`` is used. 
- Bats fly randomly to search for the prey with frequency varying between ``fmin`` and ``fmax``.
- The bats emit pulses with emission rate represented by the asymptotic value (``r0``). The value of emission rate is updated in generation ``i`` according to :math:`r_i = r_0(1-exp(-\gamma i))`, where ``gamma`` is the exponential factor of the pulse rate. ``r`` typically decreases abruptly at the beginning and then converges back to ``r0`` by the end of the evolution. 
- We provide a flexible BAT implemetation that can handle continuous (``float``), discrete (``int``), and categorical (``grid``) spaces and their mix. The user can control the type of discrete transformation via the argument ``int_transform``. 
- ``ncores`` argument evaluates the fitness of all bats in parallel. Therefore, set ``ncores <= nbats`` for most optimal resource allocation.
- Look for an optimal balance between ``nbats`` and ``ngen``, it is recommended to minimize the number of ``nbats`` to allow for more updates and more generations.
- Total number of cost evaluations for BAT is ``3*nbats`` * ``(ngen + 1)``.