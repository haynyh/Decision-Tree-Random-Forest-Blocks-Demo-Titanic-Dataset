
# Scientific and vector computation for python
import numpy as np

# Data analysis and manipulation tool for python
import pandas as pd

# Plotting library
import matplotlib.pyplot as plt 

# Unlike Assignment 1 that only uses basic libraries, 
# Assignment 2 utilizes sckit-learn libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from .tree_archit import *
from .algorithm import *



__all__ = ['generate_forest', 'generate_forest_scikit', 'forest_predict', 'forest_predict_scikit']

def generate_forest(data_train, feature_name, class_name, tree_num):
    
    # create 'forest' variable to save decision trees
    forest = list()
    feature_subset_list = list()
    
    # number of selected features for each decision tree
    feature_num = 6
    for _ in range(tree_num):
        # print('Create Tree {}'.format(num))
        # 1. randomly select subset of features from all the features.
        # 2. randomly select subset of dataset from the whole training set
        # 3. generate the decision tree with the selected features with function "generate_tree()".
        feature_subset = np.random.choice(feature_name, size=feature_num, replace=False)
        data_subset = data_train.sample(n=7180, replace = True, frac=None)
        tree_root = generate_tree(data_subset, feature_subset, class_name)
        feature_subset_list.append(feature_subset)
        forest.append(tree_root)
        
    return forest, feature_subset_list



def generate_forest_scikit(data_train, feature_name, class_name, tree_num):
    
    # create 'forest' variable to save decision trees
    forest = list()
    feature_vector = list()
    
    # number of selected features for each decision tree
    feature_num = int(np.sqrt(len(feature_name)))
    for _ in range(tree_num):
        feature_subset = np.random.choice(feature_name, size=feature_num, replace=False).tolist()
        data_subset = data_train.sample(frac=1, replace=True)
        sub_features = data_subset[feature_subset].to_dict('records')
        sub_labels = data_subset[class_name]
        
        # one-hot encoding ******Because scikit learning does not allow categorical data
        vec = DictVectorizer()
        x_data = vec.fit_transform(sub_features).toarray()
        y_data = sub_labels.values
        
        # decision tree classifier
        tree_root = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
        tree_root.fit(x_data, y_data)

        feature_vector.append(vec)
        forest.append(tree_root)
        
    return forest, feature_vector




def forest_predict(forest, feature_subset_list, query): 
    
    prediction_list = list()
    # predict label in each decision tree of forest
    for tree_node, feature_subset in zip(forest, feature_subset_list):
        label = tree_predict(tree_node, query[feature_subset])
        prediction_list.append(label)
    
    # majority voting
    # get prediction result with majority voting
    result_list = pd.Series(prediction_list)
    major_result = result_list.mode()

    return major_result[0]




def forest_predict_scikit(forest, feature_vector, query): 
    
    prediction_list = list()
    # predict label in each decision tree of forest
    for tree_node, vec in zip(forest, feature_vector):
        transformed_query = query.to_dict()
        encoded_query = vec.transform(transformed_query).toarray()
        label = tree_node.predict(encoded_query)
        prediction_list.append(label)

    result_list = pd.Series(prediction_list)
    major_result = result_list.mode()
    
    return major_result[0].item()