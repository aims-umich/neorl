.. _ex3:

Example 3: Welded-beam design
===============================

Example of solving the heavily-constrained engineering optimization problem "Welded-beam design" using NEORL with the ES algorithm tuned with Bayesian search.

Summary
--------------------

-  Algorithms: ES, Bayesian search for tuning
-  Type: Continuous, Single-objective, Constrained
-  Field: Structural Engineering

Problem Description
--------------------


The welded beam is a common engineering optimisation problem with an objective to find an optimal set of the dimensions :math:`h=x_1`, :math:`l=x_2`, :math:`t=x_3`, and :math:`b=x_4` such that the fabrication cost of the beam is minimized. This problem is a continuous optimisation problem. See the Figure below for graphical details of the beam dimensions (:math:`h, l, t, b`) to be optimised. 

.. image:: ../images/welded-beam.png
   :scale: 50 %
   :alt: alternate text
   :align: center
   
The cost of the welded beam is formulated as 

.. math::

	\min_{\vec{x}} f (\vec{x}) = 1.10471x_1^2x_2 + 0.04811x_3x_4 (14+x_2),

subject to 7 rules/constraints, the first on the shear stress (:math:`\tau`)
	
.. math::

	g_1(\vec{x}) = \tau(\vec{x}) - \tau_{max} \leq 0, 

the second on the bending stress (:math:`\sigma`)

.. math::
	
	g_2(\vec{x}) = \sigma(\vec{x}) - \sigma_{max} \leq 0,  

three side constraints
	
.. math::
	
	g_3(\vec{x}) = x_1 - x_4 \leq 0,  

	
.. math::
	
	g_4(\vec{x}) = 0.10471x_1^2 + 0.04811x_3x_4 (14+x_2) - 5 \leq 0,  

	
.. math::

	g_5(\vec{x}) = 0.125 - x_1 \leq 0,  
	
the sixth on the end deflection of the beam (:math:`\delta`)
	
.. math::
	g_6(\vec{x}) = \delta(\vec{x}) - \delta_{max} \leq 0, 
	

and the last on the buckling load on the bar (:math:`P_c`)
	
.. math::
	
	g_7(\vec{x}) = P - P_{c}(\vec{x}) \leq 0, 
	
while the range of the design variables are:

.. math::
	    \begin{split}
	         0.1 \leq x_1 \leq 2 &, \quad 0.1 \leq x_2 \leq 10, \\
	         0.1 \leq x_3 \leq 10 &, \quad 0.1 \leq x_4 \leq 2. \\
	    \end{split}

	
The derived variables and their related constants are expressed as follows:
	
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

.. literalinclude :: ../scripts/ex3_beam.py
   :language: python
 
Results
--------------------

After Bayesian hyperparameter tuning, the top 10 are 

.. code-block:: python

	----Top 10 hyperparameter sets----
	        cxpb  mu  alpha    cxmode     mutpb     score
	id                                                   
	13  0.140799  35    0.3     blend  0.110994  1.849573
	18  0.139643  37    0.3     blend  0.094496  1.925569
	25  0.341248  39    0.1  cx2point  0.197213  2.098090
	1   0.177505  32    0.3     blend  0.088050  2.144512
	20  0.100000  35    0.3     blend  0.104131  2.198990
	22  0.218197  30    0.3     blend  0.114197  2.228448
	17  0.364451  34    0.3     blend  0.102634  2.235059
	24  0.145365  42    0.3     blend  0.200532  2.292646
	19  0.100000  55    0.3     blend  0.104209  2.349494
	6   0.573142  38    0.4  cx2point  0.223231  2.349795

After re-running the problem with the best hyperparameter set, the convergence of the fitness function is shown below

.. image:: ../images/ex3_fitness.png
   :scale: 30%
   :alt: alternate text
   :align: center

while the best :math:`\vec{x} (x_1-x_4)` and :math:`y=f(x)` (minimum beam cost) are:

.. code-block:: python

	Best fitness (y) found: 1.849572817626747
	Best individual (x) found: [0.18756483308730693, 4.053366828472939, 8.731994883504612, 0.2231022567643955]