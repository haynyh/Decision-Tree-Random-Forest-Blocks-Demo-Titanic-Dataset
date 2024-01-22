
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



__all__ = [ 'entropy', 'compute_gain']



        
        

#Corss entropy on existing class
def entropy(dataSet, class_name):
    '''
    calculate the entropy of a feature used in a dataset
    dataSet - input dataset
    class_name - name of the target feature

    return:
    entropy value
    '''
    
    # stop when no dataset is available
    data_num = dataSet.shape[0]
    if data_num == 0:
        return 0

    label_value_dict = dataSet[class_name].value_counts().to_dict()
    # get numbers of each feature type 
    type_counts = np.array(list(label_value_dict.values()))
    
    ent = 0

    # 1. remove zero value
    # 2. compute the proportion of data instances having the specific feature value
    # 3. calculate an entropy element of a feature value
    # 4. add the entropy element to the total entropy value
    type_counts_no_zero = type_counts[type_counts!=0]
    for count in type_counts_no_zero:
        proportion = count / data_num
        ent_element = -proportion * np.log2(proportion)
        ent +=  ent_element


    return ent



#Information gain
def compute_gain(unionset, subsets, class_name):
        '''
        calculate gain
        unionset: data instances in a parent node
        subsets: disjoint subsets partitioned from unionSet

        return: 
        gain: gain(S, A) = entropy(S) - sum( p(S_v) * entropy(S_v) )
        '''
        
        # number of data items
        data_num = unionset.shape[0]
        impurityAfterSplit = 0
        

        impurityBeforeSplit = entropy(unionset, class_name)
        for i in range(len(subsets)):
            subset_num = subsets[i].shape[0]
            subset_entropy = entropy(subsets[i], class_name)
            weight = subset_num/data_num
            impurityAfterSplit += weight * subset_entropy
        
        #calculate total gain
        totalGain = impurityBeforeSplit - impurityAfterSplit
        
        return totalGain
    
    

