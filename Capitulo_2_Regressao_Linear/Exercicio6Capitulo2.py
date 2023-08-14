#Exercio adaptado do site
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_iris_logistic.html
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import cross_val_score

#importar a base de dados iris
#somente serão utilizadas as primeiras 2 características
iris = datasets.load_iris()
X = iris.data[:, :2]  
Y = iris.target

#Criar clasificador de regressão logistica linear
logreg = LogisticRegression(C=1e5)
logreg.fit(X, Y)

scores = cross_val_score(logreg, X, Y, cv=5)
print(scores)
print("%0.2f acuracia com desvio padrão de %0.2f" % (scores.mean(), scores.std()))


x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # tamanho da malha para o gráfico
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])


Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plotar a base de dados iris 
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Comprimento da Sepala')
plt.ylabel('Largura da Sepala')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
