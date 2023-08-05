
# -*- coding: utf-8 -*-
#Exercio adaptado do site
#https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


#regressão linear multipla

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetesDataset = datasets.load_diabetes()
print(diabetes_X.shape)
print(diabetesDataset.DESCR)

#Utilizar 3 variaveis de entrada da base de dados
diabetes_X = diabetes_X[:, [2,7,9]]

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
#bias
print(regr.intercept_)

