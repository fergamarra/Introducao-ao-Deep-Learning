import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#carregar o dataset diabetes PIMA
df = pd.read_csv('diabetes.csv')
print(df)
df.head()
df.info()
df.describe()

sc = StandardScaler()
X  = sc.fit_transform(df.drop('Outcome', axis=1))
y  = df['Outcome'].values
y_cat = to_categorical(y)
print(y_cat)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat,
                                                    random_state=22,
                                                    test_size= 0.25)

model = Sequential()
model.add(Dense(24, input_shape=(8,), activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=80, verbose=2, validation_split=0.02)

y_pred= model.predict(X_test)

y_test_class = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)

pd.Series(y_test_class).value_counts()


accuracy_score(y_test_class, y_pred_class)

print(classification_report(y_test_class, y_pred_class))

confusion_matrix(y_test_class, y_pred_class)

print(history.history.keys())


#acuracica


plt.pyplot.figure()
plt.pyplot.plot(history.history['accuracy'])
plt.pyplot.title('model accuracy')
plt.pyplot.ylabel('accuracy')
plt.pyplot.xlabel('epoch')
plt.pyplot.legend(['train'], loc='upper left')
plt.pyplot.show()

#loss


plt.pyplot.figure()
plt.pyplot.plot(history.history['loss'])
plt.pyplot.title('model loss')
plt.pyplot.ylabel('loss')
plt.pyplot.xlabel('epoch')
plt.pyplot.legend(['train'], loc='upper left')
plt.pyplot.show()


def pretty_confusion_matrix(y_true, y_pred, classes,
                            normalize=True, title=None,cmap=plt.cm.Blues):
    if not title:
       if normalize:
          title='Normalize confusion matrix'
       else:
          title='Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
       cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
       print('Normalized confusion matrix')
    else:
       print('Confusion matrix, without normalization')

    #fig, ax = plt.subplot()
    fig, ax = plt.pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest',cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    #Rotate the tick labels and set their alignment
    plt.pyplot.setp(ax.get_xticklabels(),rotation=45,ha="right",fontsize=16,
          rotation_mode="anchor")
    
    
    #Loop over data dimensions and create text annotations
    fmt='.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
      for j in range(cm.shape[1]):
         ax.text(j,i,format(cm[i,j],fmt),
              ha="center",va="center",fontsize=16,
              color="white" if cm[i,j]> thresh else"black")
    fig.tight_layout()
    return ax
    
    

    


class_names = ['0','1']    
  
pretty_confusion_matrix(y_test_class,y_pred_class, classes=class_names,title='Confusion matrix')
plt.pyplot.show()
