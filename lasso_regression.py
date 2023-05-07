import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Ridge:
    def __init__(self,params = [10**i for i in range(-10,2)]):
        self.params = params
        self.best_params = 0
        self.best_score = 0
    
    def fit(self,X,y):
        self.x_mean = X.mean()
        self.x_std = X.std()
        self.X = (X-self.x_mean)/self.x_std
        col = np.ones(X.shape[0],dtype=float)
        self.X.insert(0,'intercept',col)
        id_mat = np.identity(self.X.shape[1])
        id_mat[0][0]=0
        self.penalty = []
        self.B = []
        for i in self.params:
            self.penalty.append(i*id_mat)
            self.B.append(np.linalg.inv(self.X.T @ self.X + self.penalty[-1]) @  self.X.T @ y)
    
    def predict(self,X_test,y_test):
        X_test = (X_test-self.x_mean)/self.x_std        
        col = np.ones(X_test.shape[0],dtype=float)
        X_test.insert(0,'intercept',col)
        self.prediction = []
        self.mean_sq_err = []
        for i in range(len(self.B)):
            self.prediction.append(X_test.to_numpy() @ self.B[i])
            self.mean_sq_err.append(mean_squared_error(y_true=y_test,y_pred=self.prediction[-1]))

        k  =np.argmin(self.mean_sq_err)
        self.best_params = self.params[k]
        self.best_score = self.mean_sq_err[k]    
        return self.prediction[k]    
    
    
if __name__ == '__main__':
    df = pd.read_csv('teams.csv')    
    df.drop(['team','year','height'],axis=1,inplace= True)       
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2)
    rg = Ridge()
    rg.fit(X_train,y_train)
    print(rg.predict(X_test,y_test))
    print(rg.best_params)
    print(rg.best_score)