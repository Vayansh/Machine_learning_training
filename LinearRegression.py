import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

LR = 0.05
INTERATION = 1000

class LinearRegression:
    def __init__(self):
        self.alpha = LR
    
    def fit(self,X,y):
        self.X = X
        self.y = y
        m,n = self.X.shape
        self.thetha = np.zeros(n,dtype=int)
        for _ in range(INTERATION):
            y_pred = np.dot(self.X,self.thetha)
            cost = (np.sum((y_pred-self.y)**2))/(2*m)
            dthetha = np.dot(np.transpose(self.X),(self.y-y_pred))/m
            self.thetha = self.thetha - self.alpha*dthetha
        
        print(self.thetha.shape)    
        
    def predict(self,x):
        return np.dot(x,self.thetha)
    
if __name__ == '__main__':
    X,y = make_regression(n_samples=10000,n_features=10,random_state=42,noise=30)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
    
    # fig = plt.figure(figsize=(8,6))
    # plt.scatter(X,y,color = 'b',marker='o')
    # plt.show()
    
    clf = LinearRegression()
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(y_pred.shape)
    n_sample = y_test.shape[0]
    acc = np.sum(np.abs(y_pred - y_test))/n_sample
    
    print(acc)
    err = np.sum(np.abs(y_pred-y_test)) 
    print(err)
    print((err/y_test.shape[0]))
    