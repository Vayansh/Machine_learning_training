import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

class NaiveBayes:
    def __init__(self):
        pass
    def fit(self,X_train,y_train):
        self.n_rows,self.n_feats = X_train.shape
        
        self.classes = np.unique(y_train)
        
        self.n_classes = len(self.classes)
        
        self._mean = np.zeros((self.n_classes,self.n_feats))
        self._var = np.zeros((self.n_classes,self.n_feats))
        self._priors = np.zeros(self.n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            
            self._mean[idx,:] = np.mean(X_c,axis=0)
            self._var[idx,:] = np.var(X_c,axis = 0)
            self._priors[idx] = X_c.shape[0] / self.n_rows
        
    def predict(self,X):
        prediction = [self._predict(x) for x in X]
        return np.array(prediction)
    
    def _predict(self,x):
        posteriors = []
        
        for idx,c in enumerate(self.classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx,x)))
            posteriors.append(posterior+prior)
            
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self,idx,x):
        mean = self._mean[idx]
        var = self._var[idx]
        
        numerator = np.exp((-(x-mean)**2)/(2*var))
        demo = np.sqrt(2*np.pi*var)
        
        return numerator/demo
    

if __name__ == '__main__':
    def accuracy(y_test,pred):
        return np.sum(y_test == pred)/len(y_test)

    X,y = datasets.make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=123)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)
    
    nb = NaiveBayes()
    nb.fit(X_train,y_train)
    y_pred = nb.predict(X_test)
    print(accuracy(y_test,y_pred))
    
    
            
        
    