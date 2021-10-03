.. _ex2:

Example 2: Ackley with EA
============================

Example of solving the popular continuous optimization function "Ackley" using NEORL evolutionary algorithms.

Summary
--------------------

-  Algorithms: PSO, NES, DE 
-  Type: Continuous, Single-objective, Unconstrained
-  Field: Mathematical Optimization
 

Problem Description
--------------------

The mathematical definition of Ackley is:

.. math::

   f(\vec{x}) = 20-20exp\Big(-0.2\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}\Big)-exp\Big(\frac{1}{d}\sum_{i=1}^{d}cos(2\pi x_i)\Big) + exp(1) 

The Ackley function is continuous, non-convex and multimodal. This plot shows Ackley in two-dimensional (:math:`d=2`) form.

.. image:: ../images/ackley.png
   :scale: 75 %
   :alt: alternate text
   :align: center
   
:math:`\vec{x}` domain:
The function is usually evaluated on the hypercube :math:`x_i \in [-32, 32]`, for all :math:`i = 1, …, d`. The global minima for the Ackley function is:

.. math::

   f(\vec{x}^*)=0, \text{ at } \vec{x}^*=[0,0,...,0]
   
   
NEORL script
--------------------

The solution below is for a 8-dimensional Ackley function (:math:`d=8`)

.. literalinclude :: ../scripts/ex2_ackley.py
   :language: python
   
Results
--------------------

Result summary is below for the three methods in minimizing the Ackley function.

.. image:: ../images/ex2_fitness.png
   :scale: 30%
   :alt: alternate text
   :align: center

.. code-block:: python

	------------------------ PSO Summary --------------------------
	Best fitness (y) found: 6.384158766614689e-05
	Best individual (x) found: [-1.1202021943594622e-05, 1.3222010570577733e-05, -1.0037727362601807e-05, 9.389429054206202e-06, 2.4880207036828872e-05, 1.6872593760849828e-05,    2.076883222303575e-05, 1.458529398292857e-05]
	--------------------------------------------------------------
	------------------------ DE Summary --------------------------
	Best fitness (y) found: 0.0067943767106268815
	Best individual (x) found: [-0.0025073247154970765, 0.0020192971595931735, -0.0015127342773181872, -0.0010888556350037238, -0.0015830291353966849, -0.000743962941194097, 0.0002963358699222367, 0.002260054765774109]
	--------------------------------------------------------------
	------------------------ NES Summary --------------------------
	Best fitness (y) found: 1.5121439047582896e-06
	Best individual (x) found: [ 5.01688814e-07 -1.12353966e-07  7.64184537e-08  1.37674119e-08
	  3.66277722e-07 -5.94627000e-07  3.11206449e-08 -6.19858494e-07]
	--------------------------------------------------------------
