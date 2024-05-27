"""
Authors: Zheng Wang, John Griffiths, Andrew Clappison, Hussain Ather
Neural Mass Model fitting
module for wong-wang model
"""

import torch
from whobpyt.datatypes import AbstractParams, par
import numpy as np

class ParamsRWWABT(AbstractParams):
    
    def __init__(self, **kwargs):

        # default params here switched to use values from the fitted RWW model 
        
        param = {

            "std_in": par(0.0),  # standard deviation of the Gaussian noise
            "std_out": par(0.0),  # standard deviation of the Gaussian noise
            # Parameters for the ODEs
            # Excitatory population
            "W_E": par(1.),  # scale of the external input
            "tau_E": par(100.),  # decay time
            "gamma_E": par(0.641 / 1000.),  # other dynamic parameter (?)

            # Inhibitory population
            "W_I": par(0.7),  # scale of the external input
            "tau_I": par(10.),  # decay time
            "gamma_I": par(1. / 1000.),  # other dynamic parameter (?)

            # External input
            "I_0": par(0.2),  # external input
            "I_external": par(0.),  # external stimulation

            # Coupling parameters
            "g": par(1000, 1000, 1/np.sqrt(10)),  # global coupling (from all nodes E_j to single node E_i)
            "g_EE": par(3.5, 3.5, 1/np.sqrt(50)),  # local self excitatory feedback (from E_i to E_i)
            "g_IE": par(0.42, 0.42, 1/np.sqrt(50)),  # local inhibitory coupling (from I_i to E_i)
            "g_EI": par(0.42, 0.42, 1/np.sqrt(50)),  # local excitatory coupling (from E_i to I_i)

            "aE": par(310),
            "bE": par(125),
            "dE": par(0.16),
            "aI": par(615),
            "bI": par(177),
            "dI": par(0.087),

            # Output (BOLD signal)
            "alpha": par(0.32),
            "rho": par(0.34),
            "k1": par(2.38),
            "k2": par(2.0),
            "k3": par(0.48),  # adjust this number from 0.48 for BOLD fluctruate around zero
            "V": par(.02),
            "E0": par(0.34),
            "tau_s": par(1 / 0.65),
            "tau_f": par(1 / 0.41),
            "tau_0": par(0.98),
            "mu": par(0.5),

            # Params for aB-tau model (curr param values are those for AD cohort)
            "bAB_E": par(0.2),
            "sAB_E": par(2.3),
            "bt_E": par(-0.4),
            "st_E": par(-2.6),
            "bAB_I": par(0.2),
            "sAB_I": par(-2.5),
            "bt_I": par(0),
            "st_I": par(0)
        }

        for var in param:
            setattr(self, var, param[var])

        for var in kwargs:
            setattr(self, var, kwargs[var])

