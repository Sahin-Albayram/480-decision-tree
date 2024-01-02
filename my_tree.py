from sklearn.model_selection import train_test_split 
import pandas as pd
import seaborn as sns
import math
import numpy as np
import category_encoders as ce
from graphviz import Digraph
from sklearn.model_selection import KFold

class TreeNode:
    def __init__(self, feature=None, is_leaf=False, prediction=" ", subtrees=None):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.feature = feature
        self.subtrees = subtrees or {}



class Tree:
    def __init__(self,data,labels):
        self.entropy = self.calculate_entropy(data,labels) 
        self.used_features = []
        self.data = data

    def visualize_tree(self,node, graph=None, parent_name=None, edge_label=""): #visualization step with graphviz
        if graph is None:
            graph = Digraph(comment='myTree') #creating new graph

        node_code = str(hash(node))  # Create a unique name for each node

        if node.is_leaf: # Creating Leaf node in graph
            graph.node(node_code, label=f'Prediction: {node.prediction}', shape='box')
        else: # Creating Decision node in graph
            graph.node(node_code, label=f'Feature: {node.feature}')
        
        if parent_name is not None: # Connect it to parent
            graph.edge(parent_name, node_code, label=edge_label)

        
        if not node.is_leaf:# If not leaf then recursively call it
            for value, subtree in node.subtrees.items():
                self.visualize_tree(subtree, graph, node_code, edge_label=str(value))

        return graph



    def predict(self, instance): # Prediction
        prediction = []
        for i in range(len(instance.index)): # For data with multiple instances
            prediction.append(self.traverse_tree(self.root, instance.iloc[i])) # Append prediction
        return prediction

    def traverse_tree(self, node, instance): # Traverse tree with given instance
        if node.is_leaf: 
            return node.prediction

        feature_value = instance.iloc[node.feature] # Taking value of feature that the node check
        if feature_value in node.subtrees: # Look for whether node contain that option
            return self.traverse_tree(node.subtrees[feature_value], instance) # if it contains go next node 
        else: # if not then look data for that value of the feature and give most repeated class of that value
            filtered_rows = self.data[self.data.iloc[:, node.feature] == feature_value]
            return filtered_rows.iloc[:, -1].mode()[0]
        #data[data.iloc[:, best_feature] == value]
        
    def fit(self,data,labels): # Construct tree
        self.root = self.construct_tree(data, labels)

    def find_best_feature(self,data,labels): # Look entropy of each feature and select best one 
        
        self.feature_arr = self.create_feature_arr(data,labels) # Creation of Feature array 
        self.value_distribution(data,labels)                # Filling Feature array
        self.calculate_feature_entropy(data,labels)         # calculate each feature's entropy
        self.calculate_information_gain()                   # look for information gain but I did not used it much
        arr = self.feature_entropy.copy()                   # coppying array for sorting entropies of features
        arr.sort()
        for i in range(len(arr)):
            if self.feature_entropy.index(arr[i]) in self.used_features:    #Checking if feature is used before
                continue
            else:
                return self.feature_entropy.index(arr[i]) # IF not used then return


    def construct_tree(self,data,labels): # Construction of tree

        # Checking number of label type and number of used features if there is only one label in division or all features
        # are used as node then it is a leaf node with most repeated label (if it is only label then it will return itself of course)
        if labels.nunique() == 1 or len(self.used_features) == labels.nunique(): 
            return TreeNode(is_leaf=True, prediction=labels.mode()[0])

        best_feature = self.find_best_feature(data,labels) # Look for best feature

        if best_feature == None: # if none than all of them have been used
            return TreeNode(is_leaf=True, prediction=labels.mode()[0])

        self.used_features.append(best_feature) # append used feature
        node = TreeNode(feature=best_feature) # Create node

        unique_values = data.iloc[:, best_feature].unique() #Take unique values of that feature

        for value in unique_values:
            subset_data = data[data.iloc[:, best_feature] == value] # take part of data that contain "value"
            subset_labels = labels[data.iloc[:, best_feature] == value] # take part of label like data
            node.subtrees[value] = self.construct_tree(subset_data, subset_labels) # Send them to subtrees

        return node


    def calculate_feature_entropy(self,data,labels): # Entropy calculation of features
        feature_entropy = []
        for feature in self.feature_arr: # For each feature
            entropy = 0
            for f in feature: # For each option
                f_entropy = 0
                f_multiplier = sum(f)/sum(sum(feature))
                # print(f"ffff {f_multiplier}")
                if f_multiplier == 0:
                    continue
                for value in f: # For each class
                    multiplier = value/sum(f)
                    if multiplier == 0:
                        continue

                    f_entropy += -(multiplier * math.log2(multiplier)) 
                entropy = f_multiplier * f_entropy # total feature entropy
            feature_entropy.append(entropy)
        self.feature_entropy = feature_entropy

    def calculate_information_gain(self): # Calculation of information gain for each feature

        information_gain = []
        for feature_entropy in self.feature_entropy:
            gain = self.entropy - feature_entropy # Gain is difference
            information_gain.append(gain)
        
        self.information_gain = information_gain
        return information_gain
    

    def create_feature_arr(self,data,labels):  # Creation of feature array
        max_value = 0
        for col in data.columns:
            value = data[col].nunique()
            if max_value < value: # Taking maximum number of option as higher bound
                max_value = value
        # Create 3d array for each feature each option each class
        return np.zeros((len(data.columns),max_value,labels.nunique()),dtype=np.int16) 
    
    def value_distribution(self,data,labels): #filing Feature array

        # Check for each value in each column and increase specific part in feature array
        # for instance feature : buying price (idx = 0) , option: vhigh (idx = 0 ), class : acc (idx = 1)
        # feature array [0][0][1] +1
        if len(data.index) != len(labels.index):
            print(f"failure : {len(data.index)}, {len(labels.index)}")
        for num,col in enumerate(data.columns):
            for r_num,value in enumerate(data[col].unique()):
                for row in data[data[col] == value].index:
                    for l_num,l in enumerate(labels.unique()):
                        if labels.loc[row] == l:
                            self.feature_arr[num][r_num][l_num] +=1
                            break
        # print(self.feature_arr)
        return self.feature_arr


    
    def calculate_entropy(self,data,labels): # Total entropy of dataset
        entropy = 0
        for i in range(labels.nunique()):
            num = labels.value_counts().iloc[i]
            entropy -= num/labels.size * math.log2(num/labels.size)
        return entropy
    


def main():
    df = pd.read_csv("480-decision-tree/car_evaluation.csv")

    col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'] #Renaming columns
    df.columns = col_names

    x = df.drop(["class"],axis=1) #split data
    y = df["class"] # split label
    
    X_tv,X_test,y_tv,y_test=train_test_split(x,y,test_size=0.2, random_state=42) # train-validation-test split

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1

    train_acc = []
    valid_acc = []
    test_acc = []
    for train_idx, test_idx in kf.split(X_tv):
        X_train, X_valid = X_tv.iloc[train_idx], X_tv.iloc[test_idx]
        y_train, y_valid = y_tv.iloc[train_idx], y_tv.iloc[test_idx]


        myTree = Tree(X_train, y_train)
        myTree.fit(X_train, y_train)

        
        tree_graph = myTree.visualize_tree(myTree.root)
        tree_graph.render(f"tree_{fold}", view=False)

        y_predict_train = myTree.predict(X_train)
        y_predict_valid = myTree.predict(X_valid)
        y_predict_test = myTree.predict(X_test)

        count = 0
        for i in range(len(y_predict_train)):
            if y_predict_train[i] == y_train.iloc[i]:
                count+=1
        train_acc.append(count/len(y_predict_train))
        count = 0
        for i in range(len(y_predict_valid)):
            if y_predict_valid[i] == y_valid.iloc[i]:
                count+=1
        valid_acc.append(count/len(y_predict_valid))
        print(f"Fold {fold} - validation accuracy {count/len(y_predict_valid)}") 
        count = 0
        for i in range(len(y_predict_test)):
            if y_predict_test[i] == y_test.iloc[i]:
                count+=1
        test_acc.append(count/len(y_predict_test))
        
        
        fold += 1

    print(f"train_acc: {train_acc}")
    print(f"validation_acc: {valid_acc}")
    print(f"test_acc: {test_acc}")


if __name__ == "__main__":
    main()