.. _quickstart:

===============
Getting Started
===============

NEORL tries to follow a typical ML-like syntax used in most libraries like ``sklearn``, ``keras``, and ``stable-baselines``.

Here, we describe how to use NEORL to minimize the popular sphere function, which takes the form 

.. math::

   f(\vec{x}) = \sum_{i=1}^d x_i^2 

The Sphere function is continuous, convex and unimodal. This plot shows its two-dimensional (:math:`d=2`) form.

.. image:: ../images/spheref.png
   :scale: 75 %
   :alt: alternate text
   :align: center
   
:math:`\vec{x}` domain:
The function is usually evaluated on the hypercube :math:`x_i \in [-5.12, 5.12]`, for all :math:`i = 1, …, d`. The global minima for the sphere function is:

.. math::

   f(\vec{x}^*)=0, \text{ at } \vec{x}^*=[0,0,...,0]


Here is a quick example of how to use NEORL to minimize a 5-D (:math:`d=5`) sphere function:

.. code-block:: python

	#---------------------------------
	# Import packages
	#---------------------------------
	import numpy as np
	import matplotlib.pyplot as plt
	from neopy import DE, XNES

	#---------------------------------
	# Fitness
	#---------------------------------
	#Define the fitness function
	def FIT(individual):
		"""Sphere test objective function.
			F(x) = sum_{i=1}^d xi^2
			d=1,2,3,...
			Range: [-100,100]
			Minima: 0
		"""
		
		return -sum(x**2 for x in individual)      #-1 is used to convert minimization to maximization

	#---------------------------------
	# Parameter Space
	#---------------------------------
	#Setup the parameter space (d=5)
	nx=5
	BOUNDS={}
	for i in range(1,nx+1):
		BOUNDS['x'+str(i)]=['float', -100, 100]

	#---------------------------------
	# DE
	#---------------------------------
	de=DE(bounds=BOUNDS, fit=FIT, npop=60, mutate=0.5, recombination=0.7, ncores=1, seed=1)
	x_best, y_best, de_hist=de.evolute(ngen=100, verbose=0)

	#---------------------------------
	# NES
	#---------------------------------
	x0=[-50]*len(BOUNDS)
	amat = np.eye(len(x0))
	xnes = XNES(FIT, x0, amat, npop=40, bounds=BOUNDS, use_adasam=True, eta_bmat=0.04, eta_sigma=0.1, patience=9999, verbose=0, ncores=1)
	x_best, y_best, nes_hist=xnes.evolute(100)

	#---------------------------------
	# Plot
	#---------------------------------
	#Plot fitness for both methods
	plt.figure()
	plt.plot(-np.array(de_hist), label='DE')               #multiply by -1 to covert back to a min problem
	plt.plot(-np.array(nes_hist['fitness']), label='NES')  #multiply by -1 to covert back to a min problem
	plt.xlabel('Step')
	plt.ylabel('Fitness')
	plt.legend()
	plt.show()
	plt.show()
