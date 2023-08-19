import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
#carregar o dataset diabetes PIMA
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetesDataset = datasets.load_diabetes()
print(diabetes_X.shape)
print(diabetesDataset.DESCR)

df_diabetes = pd.DataFrame(diabetesDataset.data, columns=diabetesDataset.feature_names)
df_diabetes.head()
df_diabetes.hist(figsize=(12,10))
plt.show()

df_diabetes.info()

sc = StandardScaler()
X = sc.fit_transform(df_diabetes)
y = diabetes_y.data
y_cat = to_categorical(y)
print(y_cat.shape)

