# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 16:47:16 2022

@author: Majdi Radaideh
"""

import warnings
from abc import abstractmethod

import autograd
import autograd.numpy as anp
import numpy as np

from autograd.core import VJPNode, vspace, backward_pass
from autograd.tracer import new_box, isbox, toposort

# runs the function by making sure the calculations are traced using autograd
def run_and_trace(fun, x, *args, **kwargs):
    start_node = VJPNode.new_root(x)

    start_box = new_box(x, 0, start_node)
    out = fun(start_box, *args, **kwargs)

    return start_box, out


def calc_jacobian(start, end):

    # if the end_box is not a box - autograd can not track back
    if not isbox(end):
        return vspace(start.shape).zeros()

    # the final jacobian matrices
    jac = []

    # the backward pass is done for each objective function once
    for j in range(end.shape[1]):
        b = anp.zeros(end.shape)
        b[:, j] = 1
        n = new_box(b, 0, VJPNode.new_root(b))
        _jac = backward_pass(n, end._node)
        jac.append(_jac)

    jac = anp.stack(jac, axis=1)

    return jac

class Problem:
    """
    Superclass for each problem that is defined. It provides attributes such
    as the number of variables, number of objectives or constraints.
    Also, the lower and upper bounds are stored. If available the Pareto-front, nadir point
    and ideal point are stored.
    """

    def __init__(self, n_var=-1, n_obj=-1, n_constr=0, xl=None, xu=None, type_var=np.double, evaluation_of="auto"):
        """
        Parameters
        ----------
        n_var : int
            number of variables
        n_obj : int
            number of objectives
        n_constr : int
            number of constraints
        xl : np.array or int
            lower bounds for the variables. if integer all lower bounds are equal.
        xu : np.array or int
            upper bounds for the variable. if integer all upper bounds are equal.
        type_var : numpy type
            type of the variable to be evaluated. Can also be np.object if it is a complex data type
        """

        # number of variable for this problem
        self.n_var = n_var

        # type of the variable to be evaluated
        self.type_var = type_var

        # number of objectives
        self.n_obj = n_obj

        # number of constraints
        self.n_constr = n_constr

        # allow just an integer for xl and xu if all bounds are equal
        if n_var > 0 and isinstance(xl, int) and isinstance(xu, int):
            self.xl = xl if type(xl) is np.ndarray else np.ones(n_var) * xl
            self.xu = xu if type(xu) is np.ndarray else np.ones(n_var) * xu
        else:
            self.xl = xl
            self.xu = xu

        # the pareto front will be calculated only once and is stored here
        self._pareto_front = None

        # the pareto set of this problem
        self._pareto_set = None

        # actually defines what _evaluate is setting during the evaluation
        if evaluation_of == "auto":
            # by default F is set, and G if the problem does have constraints
            self.evaluation_of = ["F"]
            if self.n_constr > 0:
                self.evaluation_of.append("G")
        else:
            self.evaluation_of = evaluation_of

    # return the maximum objective values of the pareto front
    def nadir_point(self):
        """
        Returns
        -------
        nadir_point : np.array
            The nadir point for a multi-objective problem.
            If single-objective, it returns the best possible solution which is equal to the ideal point.
        """
        return np.max(self.pareto_front(), axis=0)

    # return the minimum values of the pareto front
    def ideal_point(self):
        """
        Returns
        -------
        ideal_point : np.array
            The ideal point for a multi-objective problem. If single-objective
            it returns the best possible solution.
        """
        return np.min(self.pareto_front(), axis=0)

    def pareto_front(self, *args, **kwargs):
        """
        Returns
        -------
        P : np.array
            The Pareto front of a given problem. It is only loaded or calculate the first time and then cached.
            For a single-objective problem only one point is returned but still in a two dimensional array.
        """
        if self._pareto_front is None:
            self._pareto_front = self._calc_pareto_front(*args, **kwargs)

        return self._pareto_front

    def pareto_set(self, *args, **kwargs):
        """
        Returns
        -------
        S : np.array
            Returns the pareto set for a problem. Points in the X space to be known to be optimal!
        """
        if self._pareto_set is None:
            self._pareto_set = self._calc_pareto_set(*args, **kwargs)

        return self._pareto_set

    def evaluate(self,
                 X,
                 *args,
                 return_values_of="auto",
                 return_as_dictionary=False,
                 **kwargs):

        """
        Evaluate the given problem.
        The function values set as defined in the function.
        The constraint values are meant to be positive if infeasible. A higher positive values means "more" infeasible".
        If they are 0 or negative, they will be considered as feasible what ever their value is.
        Parameters
        ----------
        X : np.array
            A two dimensional matrix where each row is a point to evaluate and each column a variable.
        return_as_dictionary : bool
            If this is true than only one object, a dictionary, is returned. This contains all the results
            that are defined by return_values_of. Otherwise, by default a tuple as defined is returned.
        return_values_of : list of strings
            You can provide a list of strings which defines the values that are returned. By default it is set to
            "auto" which means depending on the problem the function values or additional the constraint violation (if
            the problem has constraints) are returned. Otherwise, you can provide a list of values to be returned.
            Allowed is ["F", "CV", "G", "dF", "dG", "dCV", "hF", "hG", "hCV", "feasible"] where the d stands for
            derivative and h stands for hessian matrix.
        Returns
        -------
            A dictionary, if return_as_dictionary enabled, or a list of values as defined in return_values_of.
        """

        # make the array at least 2-d - even if only one row should be evaluated
        only_single_value = len(np.shape(X)) == 1
        X = np.atleast_2d(X)

        # check the dimensionality of the problem and the given input
        if X.shape[1] != self.n_var:
            raise Exception('Input dimension %s are not equal to n_var %s!' % (X.shape[1], self.n_var))

        # automatic return the function values and CV if it has constraints if not defined otherwise
        if type(return_values_of) == str and return_values_of == "auto":
            return_values_of = ["F"]
            if self.n_constr > 0:
                return_values_of.append("CV")

        # create the output dictionary for _evaluate to be filled
        out = {}
        for val in return_values_of:
            out[val] = None

        # all values that are set in the evaluation function
        values_not_set = [val for val in return_values_of if val not in self.evaluation_of]

        # have a look if gradients are not set and try to use autograd and calculate grading if implemented using it
        gradients_not_set = [val for val in values_not_set if val.startswith("d")]

        # if no autograd is necessary for evaluation just traditionally use the evaluation method
        if len(gradients_not_set) == 0:
            self._evaluate(X, out, *args, **kwargs)
            at_least2d(out)

        # otherwise try to use autograd to calculate the gradient for this problem
        else:

            # calculate the function value by tracing all the calculations
            root, _ = run_and_trace(self._evaluate, X, *[out])
            at_least2d(out)

            # the dictionary where the values are stored
            deriv = {}

            # if the result is calculated to be derivable
            for key, val in out.items():

                # if yes it is already a derivative
                if key.startswith("d"):
                    continue

                name = "d" + key
                is_derivable = (type(val) == autograd.numpy.numpy_boxes.ArrayBox)

                # if should be returned AND was not calculated yet AND is derivable using autograd
                if name in return_values_of and out.get(name) is None and is_derivable:

                    # calculate the jacobian matrix and set it - (ignore warnings of autograd here)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        if "h" + key not in out:
                            jac = calc_jacobian(root, val)
                        else:

                            def calc_gradient(X):
                                _out = {}
                                _root, _ = run_and_trace(self._evaluate, X, *[_out])
                                at_least2d(_out)
                                jac = calc_jacobian(_root, _out[key])
                                return jac

                            _root, jac = run_and_trace(calc_gradient, X)

                            hessian = []
                            for k in range(jac.shape[1]):
                                _hessian = calc_jacobian(_root, jac[:, k])
                                hessian.append(_hessian[:, None, ...])
                            hessian = np.concatenate(hessian, axis=1)
                            deriv["h" + key] = hessian

                        deriv[name] = jac

            # merge to the output
            out = {**out, **deriv}

            # convert back to conventional numpy arrays - no array box as return type
            for key in out.keys():
                if type(out[key]) == autograd.numpy.numpy_boxes.ArrayBox:
                    out[key] = out[key]._value

        # if constraint violation should be returned as well
        if self.n_constr == 0:
            CV = np.zeros([X.shape[0], 1])
        else:
            CV = Problem.calc_constraint_violation(out["G"])

        if "CV" in return_values_of:
            out["CV"] = CV

        # if an additional boolean flag for feasibility should be returned
        if "feasible" in return_values_of:
            out["feasible"] = (CV <= 0)

        # remove the first dimension of the output - in case input was a 1d- vector
        if only_single_value:
            for key in out.keys():
                if out[key] is not None:
                    out[key] = out[key][0, :]

        if return_as_dictionary:
            return out
        else:

            # if just a single value do not return a tuple
            if len(return_values_of) == 1:
                return out[return_values_of[0]]
            else:
                return tuple([out[val] for val in return_values_of])

    @abstractmethod
    def _evaluate(self, x, f, *args, **kwargs):
        pass

    def name(self):
        """
        Returns
        -------
        name : str
            The name of the problem. Per default it is the name of the class but it can be overridden.
        """
        return self.__class__.__name__

    def _calc_pareto_front(self, *args, **kwargs):
        """
        Method that either loads or calculates the pareto front. This is only done
        ones and the pareto front is stored.
        Returns
        -------
        pf : np.array
            Pareto front as array.
        """
        pass

    def _calc_pareto_set(self, *args, **kwargs):
        pass

    # some problem information
    def __str__(self):
        s = "# name: %s\n" % self.name()
        s += "# n_var: %s\n" % self.n_var
        s += "# n_obj: %s\n" % self.n_obj
        s += "# n_constr: %s\n" % self.n_constr
        s += "# f(xl): %s\n" % self.evaluate(self.xl)[0]
        s += "# f((xl+xu)/2): %s\n" % self.evaluate((self.xl + self.xu) / 2.0)[0]
        s += "# f(xu): %s\n" % self.evaluate(self.xu)[0]
        return s

    @staticmethod
    def calc_constraint_violation(G):
        if G is None:
            return None
        elif G.shape[1] == 0:
            return np.zeros(G.shape[0])[:, None]
        else:
            return np.sum(G * (G > 0).astype(np.float), axis=1)[:, None]


# makes all the output at least 2-d dimensional
def at_least2d(d):
    for key in d.keys():
        if len(np.shape(d[key])) == 1:
            d[key] = d[key][:, None]

class DTLZ(Problem):
    def __init__(self, n_var, n_obj, k=None):

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")

        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=0, xu=1, type_var=anp.double)

    def g1(self, X_M):
        return 100 * (self.k + anp.sum(anp.square(X_M - 0.5) - anp.cos(20 * anp.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return anp.sum(anp.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= anp.prod(anp.cos(anp.power(X_[:, :X_.shape[1] - i], alpha) * anp.pi / 2.0), axis=1)
            if i > 0:
                _f *= anp.sin(anp.power(X_[:, X_.shape[1] - i], alpha) * anp.pi / 2.0)

            f.append(_f)

        f = anp.column_stack(f)
        return f
    
class DTLZ2(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var, n_obj, **kwargs)

    def _calc_pareto_front(self, ref_dirs):
        return generic_sphere(ref_dirs)

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out["F"] = self.obj_func(X_, g, alpha=1)

def generic_sphere(ref_dirs):
    return ref_dirs / anp.tile(anp.linalg.norm(ref_dirs, axis=1)[:, None], (1, ref_dirs.shape[1]))