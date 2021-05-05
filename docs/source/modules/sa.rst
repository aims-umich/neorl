.. _sa:

.. automodule:: neorl.evolu.sa

Simulated Annealing (SA)
==================================
	
Coming Soon!

..
	A module for parallel Simulated Annealing. A Synchronous Approach with Occasional Enforcement of Best Solution. 
	
	Original paper: Onbaşoğlu, E., Özdamar, L. (2001). Parallel simulated annealing algorithms in global optimization. Journal of global optimization, 19(1), 27-50..
	
	What can you use?
	--------------------
	
	-  Multi processing: ✔️
	-  Discrete spaces: ✔️
	-  Continuous spaces: ✔️
	-  Mixed Discrete/Continuous spaces: ✔️
	
	Parameters
	----------
	
	.. autoclass:: SA
	  :members:
	  :inherited-members:
	  
	Example
	-------
	
	.. code-block:: python
	
		from neorl import PSO
		
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
		
		#setup and evolute PSO
		pso=PSO(mode='min', bounds=BOUNDS, fit=FIT, c1=2.05, c2=2.1, npar=50,
		                speed_mech='constric', ncores=1, seed=1)
		x_best, y_best, pso_hist=pso.evolute(ngen=100, verbose=1)
	
	Notes
	-----
	
	- Always try the three speed mechanisms via ``speed_mech`` when you solve any problem. 
	- Keep c1, c2 > 2.0 when using ``speed_mech='constric'``. 
	- ``speed_mech=timew`` uses a time-dependent inertia factor, where inertia ``w`` is annealed over PSO generations.
	- ``speed_mech=globw`` uses a ratio of swarm global position to local position to define inertia factor, and this factor is updated every generation.
	- Look for an optimal balance between ``npar`` and ``ngen``, it is recommended to minimize particle size to allow for more generations.
	- Total number of cost evaluations for PSO is ``npar`` * ``ngen``.