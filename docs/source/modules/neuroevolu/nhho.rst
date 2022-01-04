.. _nhho:

.. automodule:: neorl.hybrid.nhho


Neural Harris Hawks Optimization (NHHO)
=========================================

A module for the surrogate-based Harris Hawks Optimization trained by offline data-driven tri-training approach. The surrogate model used is feedforward neural networks constructed from tensorflow.

Original paper: Huang, P., Wang, H., & Jin, Y. (2021). Offline data-driven evolutionary optimization based on tri-training. Swarm and Evolutionary Computation, 60, 100800.


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ✔️
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ✔️

Parameters
----------

.. autoclass:: NHHO
  :members:
  :inherited-members:
  
Example
-------

.. literalinclude :: ../../scripts/ex_nhho.py
   :language: python

Notes
-----

- Tri-training concept uses semi-supervised learning to leverage surrogate models that approximate the real fitness function to accelerate the optimization process for expensive fitness functions. Three feedforward neural network models are trained, which are used to determine the best individual from one generation to the next, which is added to retrain the three surrogate models. The real fitness function ``fit`` is ONLY used to evaluate ``num_warmups``. Afterwards, the three neural network models are used to guide the Harris hawks optimizer.
- For ``num_warmups``, choose a reasonable value to accommodate the number of design variables ``x`` in your problem. If ``None``, the default value of warmup samples is 20 times the size of ``x``. 
- Total number of cost evaluations via the real fitness function ``fit`` for NHHO is ``num_warmups``.
- Total number of cost evaluations via the surrogate model for NHHO is ``2 * nhawks`` * ``ngen``.
- The following variables can be used in ``nn_params`` dictionary to construct the surrogate model

    +--------------------+-------------------------------------------------------------------------------------------------------------------+
    |Hyperparameter      | Description                                                                                                       |
    +====================+===================================================================================================================+
    |- num_nodes         |- List of number of nodes, e.g. [64, 32] creates two layer-network with 64 and 32 nodes (default: [100, 50, 25])   |
    |- learning_rate     |- The learning rate of Adam optimizer (default: 6e-4)                                                              |
    |- batch_size        |- The minibatch size (default: 32)                                                                                 |
    |- activation        |- Activation function type (default: ``relu``)                                                                     |
    |- test_split        |- Fraction of test data or test split  (default: 0.2)                                                              |
    |- epochs            |- Number of training epochs (default: 20)                                                                          |
    |- verbose           |- Flag to print different surrogate error to screen  (default: True)                                               |
    |- save_models       |- Flag to save the neural network models (default: True)                                                           |
    |- plot              |- Flag to generate plots for surrogate training loss and surrogate prediction accuracy (default: True)             |
    +--------------------+-------------------------------------------------------------------------------------------------------------------+