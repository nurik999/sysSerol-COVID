import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy import stats

Cov = pd.read_csv("../Data File/CovData.csv")
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



from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale
Y = Outcome

plsr = PLSRegression(n_components=2)
plsr.fit_transform(X, Y)
y = plsr.predict(X)
xloading, yloading = plsr.x_loadings_, plsr.y_loadings_
xscore, yscore = plsr.x_scores_, plsr.y_scores_

for i in range(0, len(Y)):
    if Y[i] == 1:
        plt.scatter(xscore[i, 0], xscore[i, 1], c='r')
        plt.legend('M')
    elif Y[i] == 0:
        plt.scatter(xscore[i, 0], xscore[i, 1], c='b') 
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scores Plot - PLS-DA')
plt.figure()

plt.scatter(xloading[:,0],xloading[:,1])
plt.scatter(yloading[:,0],yloading[:,1], c='r')
plt.annotate('Classification', (yloading[:, 0], yloading[:, 1]))
plt.xlabel("PC1")
plt.ylabel('PC2')
plt.title('Loadings Plot - PLS-DA')
for i, txt in enumerate(variables):
    plt.annotate(txt, (xloading[i,0], xloading[i,1]))
