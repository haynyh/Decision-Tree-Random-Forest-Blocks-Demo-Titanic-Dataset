
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
from .algorithm import *





__all__ = ['Node', 'split_attribute', 'generate_tree', 'tree_predict', 'print_tree']


class Node: #initiate Node for the tree
    def __init__(self, isLeaf, label, threshold):

        self.isLeaf = isLeaf
        self.label = label
        self.threshold = threshold
        self.children_features = list() # same value types in each attribute 
        self.children = list()
        



def split_attribute(curData, curAttributes, class_name):
    '''
    find the feature with the best gain, split the data, if feature is continuous, find the threshold
    curData - current input training data
    curAttributes - current input feature list

    return:
    best_attribute - the feature with best gain
    best_threshold - for categorical feature, None; for continuous feature, the threshold of dividing the dataset
    splitted - list of splited dataset
    '''
    splitted = []
    max_gain = -1 * float("inf")
    best_attribute = None
    best_subfeatures = list()
    best_threshold = None
    
    for attribute in curAttributes:
        # get type of this attribute
        data_type = curData[attribute].dtype
        if data_type == 'object': # data is categorical
            # the counts of data in different value types
            value_counts_dict = curData[attribute].value_counts().to_dict()
            # iterations on different value type (keys of dictionary)
            subsets = list()
            subfeatures = list()
            for value_type in value_counts_dict:
                subset_data = curData[curData[attribute] == value_type]
                subsets.append(subset_data)
                subfeatures.append(value_type)

            # 1. compute information gain of the categorial attribute
            gain_value = compute_gain(curData, subsets, class_name)

            # compare current "gain value" with the maximal gain and remove the smaller one
            if gain_value >= max_gain:
                splitted = subsets
                max_gain = gain_value
                best_attribute = attribute
                best_threshold = None
                best_subfeatures = subfeatures
                
        elif data_type == 'float64' or data_type == 'int64': # data type is continuous
            
            # 2. sort the data records by attribute values with ascending order
            # 3. get the candidate of threshold
            # 4. separate the data with the selected attribute
            # 5. save them in "subsets" as a list
            # 6. compute information gain of the continuous attribute
            sorted_data = curData.sort_values(attribute).reset_index(drop=True)
            for idx in range(sorted_data.shape[0]-1):
                # check if all values of the attribute are the same
                if sorted_data.loc[idx, attribute] != sorted_data.loc[idx+1, attribute]:
                    threshold = (sorted_data.loc[idx, attribute] + sorted_data.loc[idx+1, attribute] + 1)/2
                    greater_data = sorted_data[sorted_data[attribute] > threshold]
                    less_data = sorted_data[sorted_data[attribute] < threshold] 
                    subsets = [less_data, greater_data]
                    gain_value = compute_gain(curData, subsets, class_name)

                    # compare current "gain value" with the maximal gain and remove the smaller one
                    if gain_value >= max_gain:
                        splitted = subsets
                        max_gain = gain_value
                        best_attribute = attribute
                        best_threshold = threshold
                        best_subfeatures = list()
        else:
            raise ValueError('Data Type {} is not considered.'.format(data_type))
    return (best_attribute,best_threshold,splitted, best_subfeatures)



def generate_tree(curData, curAttributes, class_name):
    
    # get the counted values of different labels 
    class_dict = curData[class_name].value_counts().to_dict()
    
    
    if curData.shape[0] == 0:
        #exception handling, return a Node. This is a leaf node, label can be "Fail", threshold = None. 
        tree_node = Node(True, "Fail", None)

        
    elif len(class_dict) == 1:
        # 1. extract key(s) of dictionary "class_dict"
        # 2. get current label
        # 3. generate a new node as leaf
        keys = class_dict.keys()
        label = list(keys)[0]
        tree_node = Node(True, label, None)

        
    elif len(curAttributes) == 0:
        # in this situation, all sample in data belongs to different categories but no more feature for split, 
        # find the majority category and assign to majClass, return a node with majClass.
        # the node is a leaf node.
        # 4. get the major class from the dictionary "class_dict"
        # 5. generate a new node as leaf
        majClass = max(class_dict, key=class_dict.get)
        tree_node = Node(True, majClass, None)
        
        
    else:
        # the node is not a leaf node
        (best_attribute, best_threshold, splitted, best_subfeatures) = split_attribute(curData, curAttributes, class_name)
        if best_attribute is None:
            majClass = max(class_dict, key=class_dict.get)
            tree_node = Node(True, majClass, None)
        else:
            remainingAttributes = curAttributes[:]
            
            if isinstance(remainingAttributes, np.ndarray):
                # convert remainingAttributes to a Python list
                remainingAttributes = remainingAttributes.tolist()
                
            remainingAttributes.remove(best_attribute)
            tree_node = Node(False, best_attribute, best_threshold)
            # print('Branch node with attribute: {}'.format(best_attribute))
        
        # 6. build the children nodes recursively using generate_tree()
        tree_node.children = [generate_tree(subset, remainingAttributes, class_name) for subset in splitted]
        tree_node.children_features = best_subfeatures
        
    return tree_node 




def tree_predict(node, query):
    '''
    predict a sample
    node - node in decision tree
    query - list of a sample

    return:
    the predict result of a sample
    '''
    if not node.isLeaf:
        if node.threshold is None:
            # categorical
            for child, sub_feature in zip(node.children, node.children_features): # "zip": an iterator of tuples where the first item in each passed iterator is paired together:
                # check which branch this sample belongs according to the feature
                # compare the query value with the sub_feature in each iteration
                # 1. add return values when the child is a Leaf node or not

                if sub_feature == query[node.label]:
                    if child.isLeaf:
                        return child.label
                    else:
                        return tree_predict(child, query)

        else:
            # 2. there are two subclasses, children[0]: <=threshold, or children[1]: >threshold
            # you should check which one the query data belongs to then, you should assign it to the variable 'child'
            if query[node.label] <= node.threshold:
                child = node.children[0]
            else:
                child = node.children[1]
            if child.isLeaf:
                return child.label
            else:
                return tree_predict(child, query)
            
    else:
        # If the node is a leaf, return its label as the prediction
        return node.label

    # new feature
    return 'Failed'




def print_tree(node, w, indent=""):
    '''
    print each node recursively
    '''
    if not node.isLeaf:
        if node.threshold is None:
            #print discrete node
            for index, child in enumerate(node.children):
                if child.isLeaf:
                    w.write(indent + str(node.label) + " = " + str(node.children_features[index]) + " : " + str(child.label) + '\n')
                else:
                    w.write(indent + str(node.label) + " = " + str(node.children_features[index]) + " : \n")
                    print_tree(child, w, indent + "	")
        else:
            #print numerical node
            leftChild = node.children[0]
            rightChild = node.children[1]
            if leftChild.isLeaf:
                w.write(indent + str(node.label) + " <= " + str(node.threshold) + " : " + str(leftChild.label) + '\n')
            else:
                w.write(indent + str(node.label) + " <= " + str(node.threshold)+" : \n")
                print_tree(leftChild, w, indent + " ")
            if rightChild.isLeaf:
                w.write(indent + str(node.label) + " > " + str(node.threshold) + " : " + str(rightChild.label) + '\n')
            else:
                w.write(indent + str(node.label) + " > " + str(node.threshold) + " : \n")
                print_tree(rightChild , w, indent + "    ")