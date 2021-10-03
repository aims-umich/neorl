.. _ex7:

Example 7: Pressure Vessel Design
=====================================

Example of solving the constrained engineering optimization problem "Pressure vessel design" using NEORL with the HHO, ES, PESA, and BAT algorithms to demonstrate compatibility with mixed discrete-continuous space.

Summary
--------------------

-  Algorithms: HHO, ES, PESA, and BAT
-  Type: Mixed discrete-continuous, Single-objective, Constrained
-  Field: Mechanical Engineering

Problem Description
--------------------


The pressure vessel design is an engineering optimization problem with the objective to evaluate the optimal thickness of shell (:math:`T_s`), thickness of head (:math:`T_h`), inner radius (R), and length of shell (L) such that the total cost of material, forming, and welding is minimized accounting for 4 constraints. :math:`T_s` and :math:`T_h` are integer multiples of 0.0625 in., which are the available thicknesses of rolled steel plates, and R and L are continuous. The figure below shows the dimensions of the pressure vessel structure. 

.. image:: ../images/ex78_vessel_diagram.jpg
   :scale: 55 %
   :alt: alternate text
   :align: center
   
The equation for the cost of the pressure vessel is 

.. math::

	\min_{\vec{x}} f (\vec{x}) = 0.6224x_1x_3x_4 + 1.7781x_2x_3^2 + 3.1661x_1^2x_4 + 19.84x_1^2x_3,

subject to 4 constraints 
	
.. math::

	g_1 = -x_1 + 0.0193x_3 \leq 0,
	
	g_2 = -x_2 + 0.00954x_3 \leq 0,
	
	g_3 = -\pi x_3^2x_4 - \frac{4}{3} \pi x_3^3 + 1296000 \leq 0,

	g_4 = x_4 - 240 \leq 0,

where :math:`0.0625 \leq x_1 \leq 6.1875` (with step of 0.0625), :math:`0.0625 \leq x_2 \leq 6.1875` (with step of 0.0625), :math:`10 \leq x_3 \leq 200`, and :math:`10 \leq x_4 \leq 200`.

NEORL script
--------------------

.. literalinclude :: ../scripts/ex7_vessel.py
   :language: python

 
Results
--------------------

A summary of the results is shown below with the best :math:`(x_1, x_2, x_3, x_4)` and :math:`y=f(x)` (minimum vessel cost). PESA seems to be the best algorithm in this case. 

.. image:: ../images/ex7_pv_fitness.png
   :scale: 30%
   :alt: alternate text
   :align: center

.. code-block:: python

	------------------------ HHO Summary --------------------------
	Function: Vessel
	Best fitness (y) found: 6450.086928941204
	Best individual (x) found: [16.          8.         51.38667573 87.7107088 ]
	-------------------------------------------------------------- 
	------------------------ ES Summary --------------------------
	Best fitness (y) found: 7440.247037114203
	Best individual (x) found: [19, 10, 59.20709018618041, 39.15211859223507]
	--------------------------------------------------------------
	------------------------ PESA Summary --------------------------
	Best fitness (y) found: 6446.821261696037
	Best individual (x) found: [16, 8, 51.45490215425688, 87.29635265232538]
	--------------------------------------------------------------
	------------------------ BAT Summary --------------------------
	Best fitness (y) found: 6820.372175171242
	Best individual (x) found: [18.          9.         58.29066654 43.68984579]
	--------------------------------------------------------------
