# Decision-Tree-Random-Forest-Blocks-Demo-Titanic-Dataset
Demonstrate building blocks of decision tree and the ensemble into random forest for classification task. Reusable notebook and modify-friendly codes for reproduction.



## Repo structure

* [src/](src/): store Decision blocks: entropy and information gain computation block, tree nodes, tree splitting, random forest ensembling
* Random Forest Demo - Titanic Dataset: Demostration on how to work through the algorithm, reproducible codes and visualization


## Main files

* [__init__.py](__init__.py): init and import all defined functions
* [algorithm.py](algorithm.py): main python file storing entropy, information gain computation
* [tree_archit.py](tree_archit.py): main python file storing tree node creation, attribute split function, recursive tree generation and the constructed tree prediction
* [random_forest.py](random_forest.py): main python file storing tree ensembling to forest and its prediction function

## Dataset

* Titanic: Machine Learning from Disaster from [https://www.kaggle.com/datasets/cyrusttf/hong-kong-housing-price-2020-2023?resource=download](https://www.kaggle.com/competitions/titanic/data?select=test.csv)


## Quick start and reproducing training and prediction


* Follow through Random Forest Demo jupyter notebook, when applying new dataset, investigate the features and do customized data cleansing before applying tree. Making sure feature_num does not exceed the actual number of features in your dataset


## Tree Fiting

* Single Tree<br />
![image](https://github.com/haynyh/Decision-Tree-Random-Forest-Blocks-Demo-Titanic-Dataset/assets/46237598/55f63b61-1893-460f-8039-15b8bee7651a)


<br />

* Ensembled Tree<br />
![image](https://github.com/haynyh/Decision-Tree-Random-Forest-Blocks-Demo-Titanic-Dataset/assets/46237598/ee6e9d1e-5ba8-467b-a6ba-96d85ebc0bbb)

![image](https://github.com/haynyh/Decision-Tree-Random-Forest-Blocks-Demo-Titanic-Dataset/assets/46237598/dbe51c0e-8313-4253-a08a-b8f3c6e3afcb)




<br />
### Prediction Example

![image](https://github.com/haynyh/Decision-Tree-Random-Forest-Blocks-Demo-Titanic-Dataset/assets/46237598/1b019a81-2170-46a8-b9d1-16a6c8e33c51)

