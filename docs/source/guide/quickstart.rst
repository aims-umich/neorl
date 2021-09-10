.. _quickstart:

===============
Getting Started
===============

NEORL tries to follow a typical machine-learning-like syntax used in most libraries like ``sklearn`` and ``keras``.

Here, we describe how to use NEORL to minimize the popular sphere function, which takes the form 

.. math::

   f(\vec{x}) = \sum_{i=1}^d x_i^2 

The sphere function is continuous, convex and unimodal. This plot shows its two-dimensional (:math:`d=2`) form.

.. image:: ../images/spheref.png
   :scale: 75 %
   :alt: alternate text
   :align: center
   
The function is usually evaluated on the hypercube :math:`x_i \in [-5.12, 5.12]`, for all :math:`i = 1, …, d`. The global minimum for the sphere function is:

.. math::

   f(\vec{x}^*)=0, \text{ at } \vec{x}^*=[0,0,...,0]


Here is a quick example of how to use NEORL to minimize a 5-D (:math:`d=5`) sphere function:

.. literalinclude :: ../scripts/ex_quick.py
   :language: python