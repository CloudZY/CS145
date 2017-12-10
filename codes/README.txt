# CS145

# clusterRanking.py:
Libraries required:
* numpy
* ast
Description:
This module is used to rank the clusters (tweet events). It reads in the original tweet data and the clustered output from clustering module. It outputs the ranking results and plain texts for data visualization.
Input file:
'../data/out/clf_train.json'
'../data/out/cluster_result_k.csv' (k=1,2,3,4)
Output file:
'../data/out/ranking_result_k' (k=1,2,3,4)
'../data/out/plain_k' (k=1,2,3,4)

# decisionTree.py:
Libraries required:
* sklearn
* numpy
* ast
Description:
This module is used to construct, train and test decision tree model and random forest model. It takes in the training vectors file and the training label file. It will output four predicted classes for both decision tree and random forest model. Also it saves both prediction and ground truth in cm file to generate confusion matrix.
Input file:
"../data/out/clf_train.vectors", "../data/out/clf_train.labels"
Output file:
'../data/out/dt_predict_class_k.vectors'(k=1,2,3,4), '../data/out/dt_cm'
'../data/out/rf_predict_class_k.vectors'(k=1,2,3,4), '../data/out/rf_cm'

# neural_network.py:
Libraries required:
* PyTorch (torch)
Description:
A three-layer neural network class (model). It requires 3 parameters as input, the input layer size, the hidden layer size and the output layer size.

# run_nn.py:
Libraries required:
* ast
* PyTorch (torch)
* numpy
* sklearn
Description:
This program functions as constructing, training and testing neural network models. It takes a vector file and a label file as an input and generate the prediction results stored into four class files. Also it saves both prediction and ground truth in cm file to generate confusion matrix.
Input file:
'../data/out/clf_train.vectors', '../data/out/clf_train.labels'
Output file:
'../data/out/nn_predict_class_k.vectors'(k=1,2,3,4), '../data/out/nn_cm'

# plot_confusion_w_numbers.py
Libraries required:
* numpy
* sklearn
* matplotlib
* itertools
* ast
Description:
This module is for generating a visualization graph of a confusion matrix. It reads a list of ground truth and predictions as input and uses modules from sklearn and matplotlib to generate the graph with number and color standing for its rate. 
Input file:
'../data/out/nn_cm'
Output file:
'../data/pic/nn_cm.png'
