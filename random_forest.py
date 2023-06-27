from decision_tree import Decision_Tree
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class RandomForest():
    def __init__(self,n_trees = 10,min_sample_split = 2,max_depth = 100,n_features = None):
        self.min_sample = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []
        self.n_trees = n_trees
    
    def fit(self,X,y):
        for _ in range(self.n_trees):
            tree = Decision_Tree(self.min_sample,
                                 self.max_depth,
                                 self.n_features)
            
            X_sample,y_sample = self._bootstrap_sample(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)
        
    def _bootstrap_sample(self,X,y):
        n_sample = X.shape[0]
        idxs = np.random.choice(n_sample,n_sample,replace=True)
        return X[idxs],y[idxs]
    
    def _most_common_label(self,y):
        count = Counter(y)
        return count.most_common(1)[0][0]
    
    def predict(self,X):
        predictions = [tree.predict(X) for tree in self.trees]
        predictions = np.swapaxes(predictions,0,1)
        predictions = [self._most_common_label(i) for i in predictions]
        return predictions
   

if __name__ == '__main__':
    dataset = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(dataset.data,dataset.target,test_size= 0.2,random_state=1234)
    
    clf  = RandomForest(n_trees=20)
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    
    print(np.sum(prediction==y_test)/len(y_test))        