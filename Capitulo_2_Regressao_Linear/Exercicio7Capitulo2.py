import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as plt
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

#importar a base de dados iris
iris = datasets.load_iris()

print(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=7)

#Criar clasificador de regressão logistica linear
logreg = LogisticRegression(C=1e5)
logreg.fit(X_train, y_train)


y_pred = logreg.predict(X_test)

def apresentar_confusion_matrix(y_true, y_pred, classes,
                            normalize=True, title=None,cmap=cm.Blues):
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

    fig, ax = plt.pyplot.subplots()
    im = ax.imshow(cm, interpolation='nearest',cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,yticklabels=classes,
           title=title,
           ylabel='Real',
           xlabel='Valor Predito')
    
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
    
    

    


class_names = ['setosa','versicolor','virginica']    
  
apresentar_confusion_matrix(y_test,y_pred, classes=class_names,title='Matriz de Confusão')
plt.pyplot.show()


