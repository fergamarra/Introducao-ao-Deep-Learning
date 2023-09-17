import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential #Modelo Sequencial
from keras.layers import Dense #Camada Dense Fully Connected 
from keras.optimizers import SGD #Otimizador Stochastic Gradient Descent
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
dflist = []

#carregar o dataset iris
iris = load_iris()
X = iris.data[:, 0:4]
print(X.shape)
#rotulos
y = iris.target
print(np.unique(y))

encoder = LabelBinarizer()
Y = encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)


learning_rates = [ 0.1, 0.01, 0.001]

for lr in learning_rates:

    K.clear_session()

    #camadas da rede
    model = Sequential()
    model.add(Dense(5, input_shape=(4,), activation='relu'))
    model.add(Dense(3, activation='softmax'))
            
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
    history = model.fit(X_train,y_train, epochs=100, batch_size=16,  validation_data=(X_test, y_test),)
    dflist.append(pd.DataFrame(history.history, index=history.epoch))

print(history.history.keys())



historydf = pd.concat(dflist, axis=1)
print(historydf)
metrics_reported = dflist[0].columns
print(metrics_reported)
idx = pd.MultiIndex.from_product([learning_rates, metrics_reported],
                                 names=['learning_rates','metric'])

historydf.columns = idx



np.set_printoptions(suppress=True)


result = model.evaluate(X_test, y_test)


historydf.xs('val_loss', axis=1, level='metric').plot(ylim=(0,1.25), style = ["k--","k-","k:","k."])
plt.xlabel("Epocas")
plt.title("Loss")
plt.show()


historydf.xs('val_accuracy', axis=1, level='metric').plot(ylim=(0,1), style = ["k--","k-","k:","k."])
plt.title("Acuracia")
plt.xlabel("Epocas")

plt.show()
