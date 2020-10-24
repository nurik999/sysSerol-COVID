import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy import stats

Cov = pd.read_csv('CovData.csv')
Outcome = Cov['Outcome']
r, c = Cov.shape
X = Cov.iloc[:,13:c]
variables = X.head(0)
X = X.values

PCAdata = scale(X)
pca = PCA(n_components=2)
scores = pca.fit_transform(PCAdata)
print("Variance for each PC: ", pca.explained_variance_ratio_)

sns.scatterplot(y=scores[:,1], x=scores[:,0], data=Cov, hue="Outcome")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Scores Plot - PCA")

plt.figure()
loadings = pca.components_.T
plt.scatter(loadings[:,0], loadings[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Loadings Plot - PCA")

for i, txt in enumerate(variables):
    plt.annotate(txt, (loadings[i,0], loadings[i,1]))
