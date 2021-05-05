.. _ex3:

Example 3
===========

Example of solving the heavily-constrained engineering optimization problem "Welded-beam design" using NEORL with the ES algorithm tuned with Bayesian search.

Summary
--------------------

-  Algorithms: ES, Bayesian search for tuning
-  Type: Continuous, Single-objective, Constrained
-  Field: Structural Engineering

Problem Description
--------------------


The welded beam is a common engineering optimisation problem with an objective to find an optimal set of the dimensions :math:`h (x_1)`, :math:`l (x_2)`, :math:`t (x_3)`, and :math:`b (x_4)` such that the fabrication cost of the beam is minimized. This problem is a continuous optimisation problem. See the Figure below for graphical details of the beam dimensions (:math:`h, l, t, b`) to be optimised. 

.. image:: ../images/welded-beam.png
   :scale: 50 %
   :alt: alternate text
   :align: center
   
The cost of the welded beam is formulated as 

.. math::

	\min_{\vec{x}} f (\vec{x}) = 1.10471x_1^2x_2 + 0.04811x_3x_4 (14+x_2),

subject to 7 rules/constraints, the first on the shear stress ($\tau$)
	
.. math::

	g_1(\vec{x}) = \tau(\vec{x}) - \tau_{max} \leq 0, 

the second on the bending stress ($\sigma$)

.. math::
	
	g_2(\vec{x}) = \sigma(\vec{x}) - \sigma_{max} \leq 0,  

three side constraints
	
.. math::
	
	g_3(\vec{x}) = x_1 - x_4 \leq 0,  

	
.. math::
	
	g_4(\vec{x}) = 0.10471x_1^2 + 0.04811x_3x_4 (14+x_2) - 5 \leq 0,  

	
.. math::

	g_5(\vec{x}) = 0.125 - x_1 \leq 0,  
	
the sixth on the end deflection of the beam ($\delta$)
	
.. math::
	g_6(\vec{x}) = \delta(\vec{x}) - \delta_{max} \leq 0, 
	

and the last on the buckling load on the bar ($P_c$)
	
.. math::
	
	g_7(\vec{x}) = P - P_{c}(\vec{x}) \leq 0, 
	
while the range of the design variables are:

.. math::
	    \begin{split}
	         0.1 \leq x_1 \leq 2 &, \quad 0.1 \leq x_2 \leq 10, \\
	         0.1 \leq x_3 \leq 10 &, \quad 0.1 \leq x_4 \leq 2. \\
	    \end{split}

	
The derived variables and their related constants are expressed as follows \cite{coello2000use}:
	
.. math::

	\tau(\vec{x}) = \sqrt{(\tau')^2 + 2\tau' \tau'' \frac{x_2}{2R}+(\tau'')^2},
	
.. math::

	\tau' = \frac{P}{\sqrt{2}x_1x_2}, \tau''=\frac{MR}{J}, M= P (L+x_2/2),

.. math::
	
	R= \sqrt{\frac{x_2^2}{4}+\frac{(x_1+x_3)^2}{4}},

	
.. math::

	J= 2\Bigg[\sqrt{2}x_1x_2 \Bigg(\frac{x_2^2}{12} + \frac{(x_1+x_3)^2}{4} \Bigg) \Bigg],
	
.. math::

	\sigma(\vec{x}) = \frac{6PL}{x_4x_3^2},
	
.. math::
	
	\delta(\vec{x}) = \frac{4PL^3}{Ex_3^3x_4},
	
.. math::

	P_c(\vec{x}) = \frac{4.013E\sqrt{\frac{x_3^2x_4^6}{36}}}{L^2}\Bigg(1-\frac{x_3}{2L}\sqrt{\frac{E}{4G}}\Bigg),
	
.. math::

	\begin{split}
	   P &= 6000 \text{ lb} , L =14 \text{ in},  E=30\times 10^6 \text{ psi}, \\ 
	   G &= 12 \times 10^6 \text{ psi}, \\
	   \tau_{max} & =13,600 \text{ psi}, \sigma_{max} = 30,000 \text{ psi}, \delta_{max} = 0.25 \text{ in}
	\end{split}

NEORL script
--------------------

.. code-block:: python

	#---------------------------------
	# Import packages
	#---------------------------------
	import numpy as np
	np.random.seed(50)
	import matplotlib.pyplot as plt
	from math import sqrt
	from neorl.tune import BAYESTUNE
	from neorl import ES
	
	#**********************************************************
	# Part I: Original Problem
	#**********************************************************
	#Define the fitness function (for the welded beam)
	def BEAM(x):
	
	    y = 1.10471*x[0]**2*x[1]+0.04811*x[2]*x[3]*(14.0+x[1])
	
	    # parameters
	    P = 6000; L = 14; E = 30e+6; G = 12e+6;
	    t_max = 13600; s_max = 30000; d_max = 0.25;
	
	    M = P*(L+x[1]/2)
	    R = sqrt(0.25*(x[1]**2+(x[0]+x[2])**2))
	    J = 2*(sqrt(2)*x[0]*x[1]*(x[1]**2/12+0.25*(x[0]+x[2])**2));
	    P_c = (4.013*E/(6*L**2))*x[2]*x[3]**3*(1-0.25*x[2]*sqrt(E/G)/L);
	    t1 = P/(sqrt(2)*x[0]*x[1]); t2 = M*R/J;
	    t = sqrt(t1**2+t1*t2*x[1]/R+t2**2);
	    s = 6*P*L/(x[3]*x[2]**2)
	    d = 4*P*L**3/(E*x[3]*x[2]**3);
	    # Constraints
	    g1 = t-t_max; #done
	    g2 = s-s_max; #done
	    g3 = x[0]-x[3];
	    g4 = 0.10471*x[0]**2+0.04811*x[2]*x[3]*(14.0+x[1])-5.0;
	    g5 = 0.125-x[0];
	    g6 = d-d_max;
	    g7 = P-P_c; #done
	
	    g=[g1,g2,g3,g4,g5,g6,g7]
	    g_round=np.round(np.array(g),6)
	    w1=100
	    w2=100
	
	    phi=sum(max(item,0) for item in g_round)
	    viol=sum(float(num) > 0 for num in g_round)
	
	    reward = (y + (w1*phi + w2*viol))
	
	    return reward
	
	#**********************************************************
	# Part II: Setup parameter space
	#**********************************************************
	#--setup the parameter space for the welded beam
	lb=[0.1, 0.1, 0.1, 0.1]
	ub=[2.0, 10, 10, 2.0]
	d2type=['float', 'float', 'float', 'float']
	BOUNDS={}
	nx=4
	for i in range(nx):
	    BOUNDS['x'+str(i+1)]=[d2type[i], lb[i], ub[i]]
	
	#*************************************************************
	# Part III: Define fitness function for hyperparameter tuning
	#*************************************************************
	def tune_fit(cxpb, mu, alpha, cxmode, mutpb):
	
	    #--setup the ES algorithm
	    es=ES(mode='min', bounds=BOUNDS, fit=BEAM, lambda_=80, mu=mu, mutpb=mutpb, alpha=alpha,
	         cxmode=cxmode, cxpb=cxpb, ncores=1, seed=1)
	
	    #--Evolute the ES object and obtains y_best
	    #--turn off verbose for less algorithm print-out when tuning
	    x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)
	
	    return y_best #returns the best score
	
	#*************************************************************
	# Part IV: Tuning
	#*************************************************************
	#Setup the parameter space for Bayesian optimisation
	#VERY IMPORTANT: The order of these parameters MUST be similar to their order in tune_fit
	#see tune_fit
	param_grid={
	#def tune_fit(cxpb, mu, alpha, cxmode):
	'cxpb': [[0.1, 0.7],'float'],             #cxpb is first (low=0.1, high=0.8, type=float/continuous)
	'mu':   [[30, 60],'int'],                 #mu is second (low=30, high=60, type=int/discrete)
	'alpha':[[0.1, 0.2, 0.3, 0.4],'grid'],    #alpha is third (grid with fixed values, type=grid/categorical)
	'cxmode':[['blend', 'cx2point'],'grid'],
	'mutpb': [[0.05, 0.3], 'float']}  #cxmode is fourth (grid with fixed values, type=grid/categorical)
	
	#setup a bayesian tune object
	btune=BAYESTUNE(param_grid=param_grid, fit=tune_fit, ncases=30)
	#tune the parameters with method .tune
	bayesres=btune.tune(nthreads=1, csvname='bayestune.csv', verbose=True)
	
	print('----Top 10 hyperparameter sets----')
	bayesres = bayesres[bayesres['score'] >= 1] #drop the cases with scores < 1 (violates the constraints)
	bayesres = bayesres.sort_values(['score'], axis='index', ascending=True) #rank the scores from best (lowest) to worst (high)
	print(bayesres.iloc[0:10,:])   #the results are saved in dataframe and ranked from best to worst
	
	#*************************************************************
	# Part V: Rerun ES with the best hyperparameter set
	#*************************************************************
	es=ES(mode='min', bounds=BOUNDS, fit=BEAM, lambda_=80, mu=bayesres['mu'].iloc[0],
	      mutpb=bayesres['mutpb'].iloc[0], alpha=bayesres['alpha'].iloc[0],
	      cxmode=bayesres['cxmode'].iloc[0], cxpb=bayesres['cxpb'].iloc[0],
	      ncores=1, seed=1)
	
	x_best, y_best, es_hist=es.evolute(ngen=100, verbose=0)
	
	print('Best fitness (y) found:', y_best)
	print('Best individual (x) found:', x_best)
	
	#---------------------------------
	# Plot
	#---------------------------------
	#Plot fitness convergence
	plt.figure()
	plt.plot(np.array(es_hist), label='ES')
	plt.xlabel('Generation')
	plt.ylabel('Fitness')
	plt.legend()
	plt.savefig('ex3_fitness.png',format='png', dpi=300, bbox_inches="tight")
	plt.show()

 
Results
--------------------

After Bayesian hyperparameter tuning, the top 10 are 

.. code-block:: python

	----Top 10 hyperparameter sets----
	id	cxpb  mu  alpha    cxmode     mutpb     score
                                                   
	20  0.100000  30    0.4  cx2point  0.050000  1.854470
	1   0.177505  32    0.3     blend  0.088050  1.981251
	16  0.214306  60    0.4  cx2point  0.300000  2.009669
	5   0.573562  41    0.1     blend  0.054562  2.141732
	7   0.131645  53    0.2     blend  0.129494  2.195028
	17  0.700000  30    0.4  cx2point  0.050000  2.274378
	3   0.180873  48    0.4     blend  0.123485  2.276671
	4   0.243426  45    0.1     blend  0.217842  2.337914
	28  0.422938  60    0.4  cx2point  0.166513  2.368654
	21  0.686839  48    0.1  cx2point  0.279152  2.372720

After re-running the problem with the best hyperparameter set, the convergence of the fitness function is shown below

.. image:: ../images/ex3_fitness.png
   :scale: 30%
   :alt: alternate text
   :align: center

while the best :math:`\vec{x} (x_1-x_4)` and :math:`y=f(x)` (minimum beam cost) are:

.. code-block:: python

	Best fitness (y) found: 1.8544702483870839
	Best individual (x) found: [0.1994589637402763, 4.343869581792787, 9.105271242105985, 0.20702316005633725]