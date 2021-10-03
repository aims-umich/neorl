.. _ex6:

Example 6: Three-bar Truss Design
=====================================

Example of solving the constrained engineering optimization problem "Three-bar truss design" using NEORL with the BAT, GWO, and MFO algorithms.

Summary
--------------------

-  Algorithms: BAT, GWO, MFO
-  Type: Continuous, Single-objective, Constrained
-  Field: Structural Engineering

Problem Description
--------------------


The Three-bar truss design is an engineering optimization problem with the objective to evaluate the optimal cross sectional areas :math:`A_1 = A_3 (x_1)` and :math:`A_2 (x_2)` such that the volume of the statically loaded truss structure is minimized accounting for stress :math:`(\sigma)` constraints. The figure below shows the dimensions of the three-bar truss structure.

.. image:: ../images/three-bar-truss.png
   :scale: 75 %
   :alt: alternate text
   :align: center
   
The equation for the volume of the truss structure is 

.. math::

	\min_{\vec{x}} f (\vec{x}) = (2 \sqrt{2} x_1 + x_2) \times H,

subject to 3 constraints 
	
.. math::

	g_1 = \frac{\sqrt{2} x_1 + x_2}{\sqrt{2} x_1^2 + 2 x_1 x_2} P - \sigma \leq 0,
	
	g_2 = \frac{x_2}{\sqrt{2} x_1^2 + 2 x_1 x_2} P - \sigma \leq 0,
	
	g_3 = \frac{1}{x_1 + \sqrt{2} x_2} P - \sigma \leq 0,

where :math:`0 \leq x_1 \leq 1`, :math:`0 \leq x_2 \leq 1`, :math:`H = 100 cm`, :math:`P = 2 KN/cm^2`, and :math:`\sigma = 2 KN/cm^2`.

NEORL script
--------------------

.. literalinclude :: ../scripts/ex6_tbt.py
   :language: python

 
Results
--------------------

A summary of the results for the three differents methods is shown below with the best :math:`(x_1, x_2)` and :math:`y=f(x)` (minimum volume).

.. image:: ../images/TBT_fitness.png
   :scale: 30%
   :alt: alternate text
   :align: center

.. code-block:: python

	------------------------ BAT Summary --------------------------
	Best fitness (y) found: 263.90446934840577
	Best individual (x) found: [0.79190302 0.39920471]
	--------------------------------------------------------------
	------------------------ GWO Summary --------------------------
	Best fitness (y) found: 263.99180199625886
	Best individual (x) found: [0.78831222 0.41023435]
	--------------------------------------------------------------
	------------------------ MFO Summary --------------------------
	Best fitness (y) found: 263.9847325242824
	Best individual (x) found: [0.77788022 0.4396698 ]
	--------------------------------------------------------------
