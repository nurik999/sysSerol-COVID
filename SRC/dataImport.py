"""
Import Atyeo data, tensor formation, plotting raw data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""Data Import"""
CovDataUrl = 'https://raw.githubusercontent.com/meyer-lab/sysSerol-COVID/main/Data%20File/CovData.csv'
Cov = pd.read_csv(CovDataUrl)

def tensorView():
    """Tensor Formation (Patients x Receptors x Antigens)"""
    _, c = Cov.shape
    SMat = Cov.iloc[:,13:c-1:3]
    RMat = Cov.iloc[:,14:c:3]
    NMat = Cov.iloc[:,15:c:3]
    Tensor = np.stack((SMat, RMat, NMat))
    return(SMat, RMat, NMat, Tensor)


def dataPlot():
    "Plotting raw data, color coded by Convalescent/Deceased"
    Plot1 = sns.scatterplot(y="S IgG3", x="S FcRg2A", data=Cov, hue="Outcome")
    Plot1.set(yscale="log")

