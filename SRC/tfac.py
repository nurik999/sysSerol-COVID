import numpy as np
import pandas as pandas
import tensorly as tl 
import seaborn as sns
from tensorly.decomposition import parafac
from tensorly.metrics.regression import variance as tl_var
from tensorly.cp_tensor import cp_to_tensor

def R2X(reconstructed, original):
    """ Calculates R2X of two tensors. """
    return 1.0 - tl_var(reconstructed - original) / tl_var(original)

def cp_decomp(tensor, rank):
    """ Execute CP Decomposition on tensor.
    -----------------------------------------
    Input: 
        tensor: 3D data tensor
        rank: rank of decomposition (number of components)
    Output: 
        output[0] -- weights
        output[1] -- list of factor matricies
    -----------------------------------------
    Factor Matricies Order:
        output[1][0] -- antigen by component
        output[1][1] -- patient by component
        output[1][2] -- receptor by component
    -----------------------------------------
    Reconstructed tensor will be used only for R2X.
    """
    weights, factors = parafac(tensor, rank, tol=1e-15, n_iter_max=2000)
    recon_tensor = R2X(cp_to_tensor((weights, factors), mask=None), tensor)
    return factors, recon_tensor
