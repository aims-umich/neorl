.. _xnes:

.. automodule:: neorl.evolu.xnes


Exponential Natural Evolution Strategies (XNES)
===============================================

A module for the exponential natural evolution strategies with adaptive sampling. 

Original paper: Glasmachers, T., Schaul, T., Yi, S., Wierstra, D., Schmidhuber, J. (2010). Exponential natural evolution strategies. In: Proceedings of the 12th annual conference on Genetic and evolutionary computation (pp. 393-400).


What can you use?
--------------------

-  Multi processing: ✔️
-  Discrete spaces: ❌
-  Continuous spaces: ✔️
-  Mixed Discrete/Continuous spaces: ❌

Parameters
----------

.. autoclass:: XNES
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python

	from neorl import XNES
	
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
	
	xnes=XNES(mode='min', bounds=BOUNDS, fit=FIT, npop=50, eta_mu=0.9, 
	          eta_sigma=0.25, adapt_sampling=True)
	x_best, y_best, xnes_hist=xnes.evolute(ngen=100, x0=[25,25,25,25,25], verbose=1)

Notes
-----

- XNES is controlled by three major search parameters: the center of the search distribution :math:`\mu` (``mu``), the step size :math:`\sigma` (``sigma``), and the normalized transformation matrix :math:`B` (``B``) .  
- The user provides initial guess of the covariance matrix :math:`A` using the argument ``A``. XNES applies :math:`A=\sigma . B` to determine the initial step size :math:`\sigma` (``sigma``) and the initial transformation matrix :math:`B` (``B``).  
- If ``A`` is not provided, XNES starts from an identity matrix of size ``d``, i.e. ``np.eye(d)``, where ``d`` is the size of the parameter space. 
- If ``npop`` is ``None``, the following formula is used: :math:`npop = Integer\{4 + [3log(d)]\}`, where ``d`` is the size of the parameter space. 
- The center of the search distribution :math:`\mu` (``mu``) is updated by the learning rate ``eta_mu``. 
- The step size :math:`\sigma` (``sigma``) is updated by the learning rate ``eta_sigma``. If ``eta_sigma`` is ``None``, the following formula is used: :math:`eta\_sigma = \frac{3}{5} \frac{3+log(d)}{d\sqrt{d}}`, where ``d`` is the size of the parameter space.
- The normalized transformation matrix :math:`B` (``B``) is updated by the learning rate ``eta_Bmat``. If ``eta_Bmat`` is ``None``, the following formula is used: :math:`eta\_Bmat = \frac{3}{5} \frac{3+log(d)}{d\sqrt{d}}`, where ``d`` is the size of the parameter space.
- Activating the option ``adapt_sampling`` may help improving the performance of XNES. 
- Look for an optimal balance between ``npop`` and ``ngen``, it is recommended to minimize population size to allow for more generations.
- Total number of cost evaluations for XNES is ``npop`` * ``ngen``.