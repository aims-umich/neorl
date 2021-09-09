.. _fneat:

Feedforward Neuroevolution of Augmenting Topologies (FNEAT)
============================================================

Neuroevolution of Augmenting Topologies (NEAT) uses evolutionary genetic algorithms to evolve neural architectures, where the best optimized neural network is selected according to certain criteria. For NEORL, NEAT tries to build a neural network that minimizes or maximizes an objective function by following {action, state, reward} terminology of reinforcement learning. In FNEAT, genetic algorithms evolve Feedforward neural networks for optimization purposes in a reinforcement learning context.

Original paper: Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2), 99-127.

What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ❌
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
------------

.. autoclass:: neorl.hybrid.fneat.FNEAT
  :members:
  :inherited-members:

Example
-------

Train a FNEAT agent to optimize the 5-D sphere function

.. literalinclude :: ../scripts/ex_fneat.py
   :language: python

Notes
--------

- The following major hyperparameters can be changed when you define the ``config`` dictionary:
    +--------------------------------+---------------------------------------------------------------------------------------------------+
    |Hyperparameter                  | Description                                                                                       |
    +================================+===================================================================================================+
    |- pop_size                      |- The number of individuals in each generation (30)                                                |
    |- num_hidden                    |- The number of hidden nodes to add to each genome in the initial population (1)                   |
    |- elitism                       |- The number of individuals to survive from one generation to the next (1)                         |
    |- survival_threshold            |- The fraction for each species allowed to reproduce each generation(0.3)                          |
    |- min_species_size              |- The minimum number of genomes per species after reproduction (2)                                 |
    |- activation_mutate_rate        |- The probability that mutation will replace the node’s activation function (0.05)                 |
    |- aggregation_mutate_rate       |- The probability that mutation will replace the node’s aggregation function  (0.05)               |
    |- weight_mutate_rate            |- The probability that mutation will change the connection weight by adding a random value (0.5)   |
    |- bias_mutate_rate              |- The probability that mutation will change the bias of a node by adding a random value (0.7)      |
    +--------------------------------+---------------------------------------------------------------------------------------------------+
                                      
Acknowledgment
-----------------

Thanks to our fellows in NEAT-Python, as we have used their NEAT implementation to leverage our optimization classes. 

https://github.com/CodeReclaimers/neat-python