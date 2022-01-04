import pandas as pd
from scipy.interpolate import interp1d
import sys
import inspect
from pathlib import Path
import numpy as np

cpath = Path(inspect.getfile(sys.modules[__name__])).resolve().parent

def get_jminus(typ = "wtd"): #wtd, refl or abs
    """Retrieve j- function from model data files."""
    #pull and load data
    configA_path = cpath / Path("tools/microreactor_configA_flux_%s_b.csv"%typ)
    configB_path = cpath / Path("tools/microreactor_configB_flux_%s_b.csv"%typ)

    jm_configA = pd.read_csv(configA_path, index_col = 0)
    jm_configB = pd.read_csv(configB_path, index_col = 0)

    #adjust 0 angle to be inward to the reactor
    jm_configA["centers"] -= 3.6820
    jm_configB["centers"] -= 4.0677

    #center periodic functions on 0
    jm_configA["centers"][jm_configA["centers"] < -np.pi] += 2*np.pi
    jm_configB["centers"][jm_configB["centers"] < -np.pi] += 2*np.pi

    #add buffer on each side for periodic integral evaluations
    lowcopy = jm_configA.copy()
    lowcopy["centers"] -= 2*np.pi
    highcopy = jm_configA.copy()
    highcopy["centers"] += 2*np.pi
    jm_configA = pd.concat([jm_configA, lowcopy, highcopy])

    lowcopy = jm_configB.copy()
    lowcopy["centers"] -= 2*np.pi
    highcopy = jm_configB.copy()
    highcopy["centers"] += 2*np.pi
    jm_configB = pd.concat([jm_configB, lowcopy, highcopy])

    #reorder from -pi to pi
    jm_configA = jm_configA.sort_values(by="centers")
    jm_configB = jm_configB.sort_values(by="centers")

    return jm_configA, jm_configB

def get_alphas(typ = "wtd"): #wtd, refl, abs
    """read alphas from file"""
    fpath = cpath / Path("tools/microreactor_alpha_%s_gr.csv"%typ)
    return pd.read_csv(fpath, index_col = 0)

def adj_coords(thetas):
    """brings array of angles into [-np.pi, np.pi]"""
    thetas = thetas % (2*np.pi)
    thetas[thetas > np.pi] -= 2*np.pi
    return thetas

def integrate(x, y, lbnd, ubnd):
    """Integrate j- function given x & y across the bounds."""
    #make sure dealing with numpy arrays
    x, y = np.asarray(x), np.asarray(y)

    #identify portions of functions where x-blocks are completely enclosed
    compl_ind = np.where(np.logical_and(x > lbnd, x < ubnd))[0]

    if compl_ind.size == 0: #if both bounds fall within an interval
        idx = np.searchsorted(x, lbnd)
        y_lbnd_approx = (lbnd - x[idx-1])/(x[idx] - x[idx-1])*(y[idx] - y[idx-1]) + y[idx-1]
        y_ubnd_approx = (ubnd - x[idx-1])/(x[idx] - x[idx-1])*(y[idx] - y[idx-1]) + y[idx-1]
        return (ubnd - lbnd)*(y_ubnd_approx + y_lbnd_approx)/2

    #integrate fully-inclosed blocks
    full_blocks_integral = np.trapz(y[compl_ind], x[compl_ind])

    #integrate lower hanging partial block
    lidx = compl_ind.min()
    y_lbnd_approx = (lbnd - x[lidx-1])/(x[lidx] - x[lidx-1])*(y[lidx] - y[lidx-1]) + y[lidx-1]
    lower_block_integral = (x[lidx] - lbnd)*(y[lidx] + y_lbnd_approx)/2

    #integrate upper hanging partial block
    uidx = compl_ind.max()+1
    y_ubnd_approx = (ubnd - x[uidx-1])/(x[uidx] - x[uidx-1])*(y[uidx] - y[uidx-1]) + y[uidx-1]
    upper_block_integral = (ubnd - x[uidx - 1])*(y[uidx - 1] + y_ubnd_approx)/2

    return full_blocks_integral + lower_block_integral + upper_block_integral

def integrate_sq(x, y, lbnd, ubnd):
    """Integrate j-^2 function given x & y across the bounds. y will be squared within the routine"""
    #make sure dealing with numpy arrays
    x, y = np.asarray(x), np.asarray(y)

    #identify portions of functions where x-blocks are completely enclosed
    compl_ind = np.where(np.logical_and(x > lbnd, x < ubnd))[0]

    if compl_ind.size == 0: #if both bounds fall within an interval
        idx = np.searchsorted(x, lbnd)
        y_lbnd_approx = (lbnd - x[idx-1])/(x[idx] - x[idx-1])*(y[idx] - y[idx-1]) + y[idx-1]
        y_ubnd_approx = (ubnd - x[idx-1])/(x[idx] - x[idx-1])*(y[idx] - y[idx-1]) + y[idx-1]
        m = (y_ubnd_approx - y_lbnd_approx)/(ubnd - lbnd)
        return m**2/3*(ubnd**3 - lbnd**3) + (m*y_lbnd_approx - m**2*lbnd)*(ubnd**2 - lbnd**2)\
                + (m*lbnd - y_lbnd_approx)**2*(ubnd - lbnd)

    #integrate fully-inclosed blocks
    dx = np.diff(x[compl_ind]) #xi+1 - xi
    dx2 = np.diff(x[compl_ind]**2) #xi+1**2 - xi**2
    dx3 = np.diff(x[compl_ind]**3) #xi+1**3 - xi**3
    ms = np.diff(y[compl_ind])/dx
    xis = x[compl_ind[:-1]]
    yis = y[compl_ind[:-1]]
    full_blocks_integral = (ms**2*dx3/3+(ms*yis-ms**2*xis)*dx2+(ms*xis-yis)**2*dx).sum()

    #integrate lower hanging partial block
    lidx = compl_ind.min()
    y_lbnd_approx = (lbnd - x[lidx-1])/(x[lidx] - x[lidx-1])*(y[lidx] - y[lidx-1]) + y[lidx-1]
    m = (y[lidx] - y_lbnd_approx)/(x[lidx] - lbnd)
    lower_block_integral = m**2/3*(x[lidx]**3 - lbnd**3) \
            + (m*y_lbnd_approx-m**2*lbnd)*(x[lidx]**2 - lbnd**2) \
            + (m*lbnd - y_lbnd_approx)**2*(x[lidx] - lbnd)

    #integrate upper hanging partial block
    uidx = compl_ind.max()
    y_ubnd_approx = (ubnd - x[uidx])/(x[uidx+1] - x[uidx])*(y[uidx+1] - y[uidx]) + y[uidx]
    m = (y_ubnd_approx - y[uidx])/(ubnd - x[uidx])
    upper_block_integral = m**2/3*(ubnd**3 - x[uidx]**3) \
            + (m*y[uidx]-m**2*x[uidx])*(ubnd**2 - x[uidx]**2) \
            + (m*x[uidx] - y[uidx])**2*(ubnd - x[uidx])

    return full_blocks_integral + lower_block_integral + upper_block_integral

def int_bounds(theta, cangle):
    """get bounds on j-^2 integrals given rotation angle, theta and coating angle cangle
    returns (thetaA&thetaA0'), (thetaA'&thetaA0)"""
    assert (theta < np.pi + 1e-5) and (theta > -np.pi - 1e-5)

    if 0 < theta and theta < cangle:
        return ([cangle/2, theta + cangle/2], [-cangle/2, theta - cangle/2])
    elif -cangle < theta and theta < 0:
        return ([theta - cangle/2, -cangle/2], [theta + cangle/2, cangle/2])
    else:
        return ([theta - cangle/2, theta + cangle/2], [-cangle/2, cangle/2])

def calc_zetatildes(theta, cangles, alphas, jminusA, jminusB):
    """
    calculate zetatilde functions for all drums
    thetas is numpy array of size 8 of rotation angles
    cangles is numpy array of size 8 of coating angles
    alphas is numpy array of shape (8,9) of model coefficients
    jminusA is pandas array with "centers" and "hist" columns
    jminusB is pandas array with "centers" and "hist" columns
    """
    configAids = [1, 4, 5, 8] #drum positions with configuration A

    #calculate gammastar by integrting j- over abs bounds for each drum
    gammastar = np.ones(9)
    for i in range(1, 9):
        if i in configAids:
            gammastar[i] = 1 - integrate(x = jminusA["centers"],
                                         y = jminusA["hist"],
                                         lbnd = theta[i-1] - cangles[i-1]/2,
                                         ubnd = theta[i-1] + cangles[i-1]/2)
        else:
            gammastar[i] = 1 - integrate(x = jminusB["centers"],
                                         y = jminusB["hist"],
                                         lbnd = theta[i-1] - cangles[i-1]/2,
                                         ubnd = theta[i-1] + cangles[i-1]/2)

    #calculate zetatilde for each drum
    return (gammastar@alphas.T).values

def calc_dzetatildes(thetas, k, cangles, alphas, jmfs):
    """
    calculate derivative zetatilde functions for all drums resct to drum k
    thetas is numpy array of size 8 of rotation angles
    k is which of drums angle is taken respect to (1-indexed)
    cangles is numpy array of size 8 of coating angles
    alphas is numpy array of shape (8,9) of model coefficients
    jmfs is list of functions for inward current
    return is 8 element array
    """
    alpha_slice = alphas["alpha" + str(k)].values
    dgamma = -jmfs[k-1](thetas[k-1] + cangles[k-1]/2) + jmfs[k-1](thetas[k-1] - cangles[k-1]/2)
    return alpha_slice*dgamma

def calc_W(theta, cangle, jmf):
    """
    Calculate derivative of difference of integrals
    theta is k-th drum angle
    cangles is k-th drum coating angle
    jmf is jminus of k-th rum
    """
    t1 = jmf(theta + cangle/2)**2
    t2 = jmf(theta - cangle/2)**2
    return t1 - t2

class ReactivityModel:
    """
    Used to evaluate reactivity insertion from control drum perturbation.
    Set up as init->method call to minimize file reading times
    """
    def __init__(self, typ = "wtd"): #abs, wtd or refl
        """initialize to perform all file I/O"""
        self.jmA, self.jmB = get_jminus(typ)
        self.alphas = get_alphas(typ)
        self.cangles = np.array([130, 145, 145, 130,
                                 130, 145, 145, 130])/180*np.pi
        fA = interp1d(self.jmA["centers"].values, self.jmA["hist"].values)
        fB = interp1d(self.jmB["centers"].values, self.jmB["hist"].values)
        self.jmfs = [fA, fB, fB, fA, fA, fB, fB, fA]

    def eval(self, pert, nom = None):
        """
        Evaluate reactivity worth of drum perturbation.
        Pert is numpy array of 8 drum angles in radians with 
        coordinate systems described in the README.md.
        Nom is an optional starting state given same as pert
        """
        #bring drum angles into [-np.pi, np.pi]
        pert = adj_coords(pert)

        #pull in zetatildes from external function
        zetatildes = calc_zetatildes(theta = pert,
                                     cangles = self.cangles,
                                     alphas = self.alphas,
                                     jminusA = self.jmA,
                                     jminusB = self.jmB)

        #loop through drums and calculate each contribution
        reactivities = np.zeros(8)
        for i in range(8):
            b1, b2 = int_bounds(pert[i], self.cangles[i])
            if i in [0, 3, 4, 7]: #if config A
                int1 = integrate_sq(self.jmA["centers"], self.jmA["hist"], *b1)
                int2 = integrate_sq(self.jmA["centers"], self.jmA["hist"], *b2)
            else: #if config A
                int1 = integrate_sq(self.jmB["centers"], self.jmB["hist"], *b1)
                int2 = integrate_sq(self.jmB["centers"], self.jmB["hist"], *b2)
            reactivities[i] = zetatildes[i]*(int1 - int2)

        #basically just return conditionals
        if nom: #little trick
            return self.eval(pert) - self.eval(nom) #assume
                                                          #reactivites additive
        else:
            return reactivities.sum()

    def evald(self, pert, k, zetatildes = None):
        """
        Evaluate differential reactivity worth of drum config from single drum.
        Pert is numpy array of 8 drum angles in radians with 
        k is which drum rotation to take derivative respect to
        zetatildes should be ignored, used for computational efficiency
        coordinate systems described in the README.md.
        """
        #bring drum angles into [-np.pi, np.pi]
        pert = adj_coords(pert)

        #calculate zetatilde parameters
        dzetatildes = calc_dzetatildes(thetas = pert,
                                      k = k,
                                      cangles = self.cangles,
                                      alphas = self.alphas,
                                      jmfs = self.jmfs)
        if zetatildes is None:
            zetatildes = calc_zetatildes(theta = pert,
                                         cangles = self.cangles,
                                         alphas = self.alphas,
                                         jminusA = self.jmA,
                                         jminusB = self.jmB)

        #define vector to hold summation
        drdtk = np.zeros(8)

        #iterate through all drums
        for i in range(8):
            b1, b2 = int_bounds(pert[i], self.cangles[i])
            if i in [0, 3, 4, 7]: #if config A
                int1 = integrate_sq(self.jmA["centers"], self.jmA["hist"], *b1)
                int2 = integrate_sq(self.jmA["centers"], self.jmA["hist"], *b2)
            else: #if config B
                int1 = integrate_sq(self.jmB["centers"], self.jmB["hist"], *b1)
                int2 = integrate_sq(self.jmB["centers"], self.jmB["hist"], *b2)

            drdtk[i] += dzetatildes[i] * (int1 - int2)

        #tack on extra term from eq41
        drdtk[k-1] += zetatildes[k-1]*calc_W(pert[k-1], self.cangles[k-1], self.jmfs[k-1])

        return drdtk.sum()

    def evalg(self, pert):
        """
        Evaluate differential reactivity worth of drum config for all drums.
        Equivalent to the gradient.
        Pert is numpy array of 8 drum angles in radians with 
        coordinate systems described in the README.md.
        """
        zetatildes = calc_zetatildes(theta = pert,
                                     cangles = self.cangles,
                                     alphas = self.alphas,
                                     jminusA = self.jmA,
                                     jminusB = self.jmB)
        grad = np.zeros(8)
        for i in range(8):
            grad[i] = self.evald(pert, i+1, zetatildes)
        return grad

def reactivityModelEval(pert, nom = None, typ = "wtd"):
    """Wrapper for ReactivityModel that initializes and runs"""
    a = ReactivityModel(typ)
    return a.eval(pert, nom)

def reactivityModelEvald(pert, k, typ = "wtd"):
    """Wrapper for ReactivityModel that initializes and runs"""
    aa = ReactivityModel(typ)
    return aa.evald(pert, k)

def reactivityModelEvalg(pert, typ = "wtd"):
    """Wrapper for ReactivityModel that initializes and runs"""
    aa = ReactivityModel(typ)
    return aa.evalg(pert)

if __name__ == "__main__":
    a = ReactivityModel()
    p5 = np.array([0, 0, 2.201331,-2.137274,-2.512146,2.02961,0.19566370000000002,3.140272])
    print(a.eval(p5))
