.. _pesa:

.. automodule:: neorl.hybrid.pesa


Prioritized replay Evolutionary and Swarm Algorithm (PESA)
===========================================================

A module for the parallel hybrid PESA algorithm with prioritized experience replay from reinforcement learning. This is the classical PESA that hybridizes PSO, ES, and SA modules within NEORL.

Original paper: Radaideh, M. I., Shirvan, K. (2021). Prioritized Experience Replay for Parallel Hybrid Evolutionary and Swarm Algorithms: Application to Nuclear Fuel, *Under Review*.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: PESA
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../../scripts/ex_pesa.py
   :language: python

Notes
-----
- PESA is symmetric, meaning population size is equal between PSO, ES, and SA, which is helpful to ensure that all algorithms can update the memory with similar computing time. For example, if the user sets ``npop=60``, then in every generation, the swarm of PSO has 60 particles, ES population has 60 individuals, and SA chain has length of 60.   
- ``mu`` defines the number of individuals from ``npop`` to survive to the next generation, and also the number of samples to replay from the memory. This is applicable to PSO and ES alone as SA does not have the concept of population. For example, by setting ``mu=40`` and ``npop=60``, then after every generation, the top 40 individuals in PSO and ES survive. Then the replay memory feeds 40 individuals to ES, which form the new pool of 80 individuals that go through offspring processes that produce again ``npop=60``. For PSO, the replay memory provides ``60-40=20`` particles to form a new swarm of ``npop=60``.   
- For complex problems and limited memory, we recommend to set ``memory_size ~ 5000``. When the memory gets full, old samples are overwritten by new ones. Allowing a large memory for complex problems may slow down PESA as handling large memories is more computationally exhaustive. If ``memory_size = None``, the memory size will be set to maximum value of ``ngen*npop*3``.
- For parallel computing of PESA, pick ``ncores`` divisible by 3 (e.g. 6, 18, 30) to ensure equal computing power across the internal algorithms.  
- If ``ncores=1``, serial calculation of PESA is executed.  
- Example on how to assign computing resources for PESA. Lets assume a generation of ``npop=60`` individuals and ``ncores=30``. Then, ``icores=int(ncores/3)`` or ``icores=10`` cores are assigned to each algorithm of PSO, ES, and SA. In this case, the ES population has size ``npop=60`` and it is evaluated in parallel with 10 cores. PSO swarm also has ``npop=60`` particles evaluated with 10 cores. SA releases 10 parallel chains, each chain evaluates 6 individuals.
- Check the sections of :ref:`PSO <pso>`, :ref:`ES <es>`, and :ref:`SA <sa>` for notes on the internal algorithms and the auxiliary parameters of PESA.
- Start the prioritized replay with a small fraction for ``alpha_init < 0.1`` to increase randomness earlier to improve PESA exploration. Choose a high fraction for ``alpha_end > 0.9`` to increase exploitation by the end of evolution.    
- The rate of ``alpha_backdoor`` replaces the regular random-walk sample of SA with the best individual in the replay memory to keep SA chain up-to-date. For example, ``alpha_backdoor=0.1`` implies that out of 10 individuals in the SA chain, 1 comes from the memory and the other 9 come from classical random-walk. Keep the value of ``alpha_backdoor`` small enough, e.g. ``alpha_backdoor < 0.2``, to avoid SA divergence. 
- Look for an optimal balance between ``npop`` and ``ngen``, it is recommended to minimize population size to allow for more generations.
- Total number of cost evaluations for PESA is ``ngen*npop*3 + warmup``.