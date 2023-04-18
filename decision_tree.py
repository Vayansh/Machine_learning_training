import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self,feature = None, threshold = None, left = None,right = None,*,value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
     

class Decision_Tree:
    def __init__(self,min_sample_split = 2,max_depth = 100,n_features = None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        
    def fit(self,X,y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features,X.sample[1])
        self.root = self._grow_tree(X,y,0)
        
    def _grow_tree(self,X,y,depth):
            n_samples,n_feats = X.shape
            n_labels = len(np.unique(y))
            
        #   check stopping creteria
            if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample_split):
                leaf_value = self._most_common_label(y)
                return Node(value= leaf_value)         
        #   find best fit

            feat_idx = np.random.choice(n_feats,self.n_features,replace=False)
            best_feats,best_threshold  = self._best_spit(X,y,feat_idx)
        
        #   children nodes  
            l_idxs,r_idxs = self._split(X[:,best_feats],best_threshold)
            left = self._grow_tree(X[l_idxs,:],y[l_idxs],depth+1)
            right = self._grow_tree(X[r_idxs,:],y[r_idxs],depth+1)
            
            return Node(best_feats,best_threshold,left,right)                        
            
    
    def _most_common_label(self,y):
        count = Counter(y)
        return count.most_common(1)[0][0]
            
            
    def _best_spit(self,X,y,feat_idx):
        best_gain = -1
        best_feat,best_threshold = None,None
        for feat_id in feat_idx:
            X_col = X[:,feat_id]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._information_gain(y,X_col,threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_id
                    best_threshold = threshold
                    
        return best_feat,best_threshold
    
    def _information_gain(self,y,X_col,threshold):
        # parent entropy
        p_e = self._entropy(y)
        
        # create children
        left_idxs,right_idxs = self._split(X_col,threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 
        
        # calculate weighted average of children 
        n = len(y)
        n_l, n_r = len(left_idxs),len(right_idxs)
        e_l , e_r = self._entropy(y[left_idxs]) , self._entropy(y[right_idxs])
        
        
        # return information gain
        return p_e - ((n_l*e_l + n_r*e_r)/n)
        
    def _split(self,X_col,threshold):
        l_idx = np.argwhere(X_col <= threshold).flatten()
        r_idx = np.argwhere(X_col > threshold).flatten()
        return l_idx,r_idx
    
    def _entropy(self,y):
        hist = np.bincount(y)  # counts frequency of every unique element 
        prob = hist/len(y)       
        return -np.sum([pr*np.log(pr) for pr in prob if pr>0])
            
    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x,node.left)
        else:
            return self._traverse_tree(x,node.right) 
        
if __name__ == '__main__':
    dataset = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(dataset.data,dataset.target,test_size= 0.2,random_state=1234)
    
    clf  = Decision_Tree(max_depth=7)
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    
    print(np.sum(prediction==y_test)/len(y_test))
    