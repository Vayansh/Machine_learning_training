import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
 


class KNN:
    def __init__(self,k):
        self.k = k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        prediction = [self._predict(x) for x in X]
        return prediction

    def eucludian_distance(self,x1,x2):
        distance = np.sqrt(np.sum((x1-x2)**2))
        return distance
    
    def _predict(self,x):
        distances = [self.eucludian_distance(x,x_train) for x_train in self.X_train]
        
        k_indexes = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indexes]
        
        
        most_common = Counter(k_nearest_labels).most_common()

        return most_common[0][0]
    
if __name__ == '__main__':
    iris = datasets.load_iris()
    X,y = iris.data , iris.target
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
    
    cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])
    plt.figure()
    plt.scatter(X[:,2],X[:,3],c=y,cmap= cmap,s=20)
    plt.show()       
    
    clf = KNN(k=5)
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    acc = np.sum(prediction==y_test)/len(y_test)
    print(acc)
    
    