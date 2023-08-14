#Exercio adaptado do site
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.inspection   import DecisionBoundaryDisplay 
from sklearn import datasets

#importar a base de dados iris
#somente serão utilizadas as primeiras 2 características
iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target
print(Y)
print(Y.shape)

#Criar clasificador de regressão logistica linear
logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)

_, ax = plt.subplots(figsize=(4, 3))

DecisionBoundaryDisplay.from_estimator(
    logreg,
    X,
    cmap=plt.cm.Greys,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
    xlabel="Comprimento da sepala",
    ylabel="largura da sepala",
    eps=0.5,
)

# Plotar a base de dados iris 
plt.scatter(X[:, 0], X[:, 1], c=Y,alpha =1 , linewidths = 2, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Comprimento da Sepala')
plt.ylabel('Largura da Sepala')

plt.xticks(())
plt.yticks(())

plt.show()
