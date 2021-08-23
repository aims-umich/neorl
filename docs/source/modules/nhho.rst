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

.. code-block:: python

	from neorl import NHHO
	
	#Define the fitness function
	def FIT(individual):
	    """Sphere test objective function.
	                    F(x) = sum_{i=1}^d xi^2
	                    d=1,2,3,...
	                    Range: [-100,100]
	                    Minima: 0
	    """
	    y=sum(x**2 for x in individual)
	    return y
	
	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
	    BOUNDS['x'+str(i)]=['float', -100, 100]
	
	nn_params = {}
	nn_params['num_nodes'] = [60, 30, 15]
	nn_params['learning_rate'] = 8e-4
	nn_params['epochs'] = 100
	nn_params['plot'] = False #will accelerate training
	nn_params['verbose'] = False #will accelerate training
	nn_params['save_models'] = False  #will accelerate training
	
	nhho = NHHO(mode='min', bounds=BOUNDS, fit=FIT, nhawks=20, nn_params=nn_params, seed=1)
	individuals, fitnesses = nhho.evolute(ngen=25, verbose=True)
	
	#make evaluation of the best individuals using the real fitness function
	real_fit=[FIT(item) for item in individuals]
	
	#print the best individuals/fitness found
	min_index=real_fit.index(min(real_fit))
	print('------------------------ Final Summary --------------------------')
	print('Best real individual:', individuals[min_index])
	print('Best real fitness:', real_fit[min_index])
	print('-----------------------------------------------------------------')

Notes
-----

- Tri-training concept uses semi-supervised learning to leverage surrogate models that approximate the real fitness function to accelerate the optimization process for expensive fitness functions. Three feedforward neural network models are trained, which are used to determine the best individual from one generation to the next, which is added to retrain the three surrogate models. The real fitness function ``fit`` is ONLY used to evaluate ``num_warmups``. Afterwards, the three neural network models are used to guide the Harris hawks optimizer.
- For ``num_warmups``, choose a reasonable value to accommodate the number of design variables ``x`` in your problem. If ``None``, the default value of warmup samples is 20 times the size of ``x``. 
- Total number of cost evaluations via the real fitness function ``fit`` for NHHO is ``num_warmups``.
- Total number of cost evaluations via the surrogate model for NHHO is ``nhawks`` * ``ngen``.
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