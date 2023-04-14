import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def sigmoid(linear_pred):
    return 1/(1+np.exp(-linear_pred))


class LogisticRegression():
    
    def __init__(self,lr = 0.001,n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.bais = None
        self.weights = None
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bais = 0
        
        for _ in range(self.n_iter):
            linear_pred = np.dot(X,self.weights)
            predictions = sigmoid(linear_pred)
            
            dw = (1/n_samples)*2*np.dot(X.T,(predictions-y))
            db =  (1/n_samples) *np.sum(2*(predictions-y))
            
            self.weights = self.weights - self.lr * dw
            self.bais = self.bais - self.lr * db
            
    
    def predict(self,X):
        linear_pred = np.dot(X,self.weights)
        predictions = sigmoid(linear_pred)

        class_pred = [0 if y<=0.5 else 1 for y in predictions]
        
        return class_pred
    
    
if __name__ =='__main__':
    df = load_breast_cancer()
    X,y = df.data , df.target

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.2,random_state=42)

    clf = LogisticRegression()
    clf.fit(X_train,y_train)

    pred = clf.predict(X_test)

    print(np.sum(pred == y_test)/len(y_test))
