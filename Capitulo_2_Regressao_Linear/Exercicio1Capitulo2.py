# -*- coding: utf-8 -*-
#Exercio adaptado do site
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Carregar a base de dados diabetes
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetesDataset = datasets.load_diabetes()
print(diabetes_X.shape)
print(diabetesDataset.DESCR)

#Utilizar uma única característica
#IMC (Indice de massa corporal) na entrada
diabetes_X = diabetes_X[:, np.newaxis, 2]

print(diabetes_X[1])
print(diabetes_X.view)

#plotar as variáveis
plt.figure()
plt.title('Indice de Massa Corporal vs Avanço da doença')
plt.xlabel('Indice de Massa Corporal')
plt.ylabel('Avanço da Doença')
plt.plot(diabetes_X, diabetes_y, 'k.')
plt.grid(True)
plt.show()


# Dividir a data de entrada em subconjuntos de treinamento/teste
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Dividir a data de saída em subconjuntos de treinamento/teste
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Criar a função de regressão linear
regr = linear_model.LinearRegression()

# Treinar o modelo utilizando subconjunto de treinamento
regr.fit(diabetes_X_train, diabetes_y_train)

# Fazer as predições utilizando o subconjunto de teste
diabetes_y_pred = regr.predict(diabetes_X_test)

regr.coef_
#Coeficiente de regressão
print('Coefficients: \n', regr.coef_)
print(regr.intercept_)


# Plotand a saída do modelo
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
