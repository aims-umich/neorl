.. _cs:

.. automodule:: neorl.evolu.cs


Cuckoo Search (CS)
===============================================

A module for the Cuckoo Search Algorithm with parallel computing support. 

Original paper: Yang, X. S., & Deb, S. (2009, December). Cuckoo search via Lévy flights. In 2009 World congress on nature & biologically inspired computing (NaBIC) (pp. 210-214). IEEE.

.. image:: ../images/cuckoo.jpg
   :scale: 80%
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

.. autoclass:: CS
  :members:
  :inherited-members:
  
Example
-------

.. code-block:: python
	
	from neorl import CS
	
	#Define the fitness function
	def Sphere(individual):
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
	
	#setup and evolute CS
	cs = CS(mode = 'min', bounds = BOUNDS, fit = Sphere, ncuckoos = 40, pa = 0.25, seed=1)
	x_best, y_best, cs_hist=cs.evolute(ngen = 200, verbose=True)

Notes
-----

- CS algorithm is based on the obligate brood parasitic behavior of some cuckoo species in combination with the Levy flight behavior of some birds and fruit flies. CS assumes each cuckoo lays one egg at a time, and dump its egg in randomly chosen nest. Also, the best nests with high quality of eggs will carry over to the next generations.
- ``pa`` controls exploration/exploitation of the algorithm, it is the fraction of the cuckoos/nests that will be replaced by new cuckoos/nests. In this case, the host bird can either throw the egg away or abandon the nest, and build a completely new nest. 
- ``ncores`` argument evaluates the fitness of all cuckoos in the population in parallel. Therefore, set ``ncores <= ncuckoos`` for most optimal resource allocation.
- Look for an optimal balance between ``ncuckoos`` and ``ngen``, it is recommended to minimize the number of ``ncuckoos`` to allow for more updates and more generations.
- Total number of cost evaluations for CS is ``2*ncuckoos`` * ``ngen``.