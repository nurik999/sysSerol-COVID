import numpy as np
import seaborn as sns
from ..dataImport import tensorView
from ..tfac import cp_decomp
from .figureCommon import subplotLabel, getSetup

S, R, N, tensor = tensorView()
component = 5

def makeFigure():
    """Get a list of axis objects and create a figure."""
    row = 1
    col = 1
    ax, f = getSetup((11, 11), (row, col))
    R2X_plot(tensor, ax[0])
    subplotLabel(ax)
    return f

def R2X_plot(tensor, ax):
    """Creates R2X plot for tensor input"""
    cp_r2x = np.zeros(component)
    comps = range(1, len(cp_r2x))
    for i in comps:
        cp_r2x[i] = cp_decomp(tensor, i)[1]
    sns.scatterplot(np.arange(component), cp_r2x, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel("R2X")
    ax.set_title("CP Decomposition")
    ax.set_yticks([0, .2, .4, .6, .8, 1])