# CS145

# feature_vector_generator.py
Libraries required:
* nltk
* sklearn
* numpy
Description:
This module reads from file crawled json data file and labels, extracts features and generates feature vectors for each data. It uses SVM to filter out non event noise data and save feature vectors, json and labels for event data for classification modules to use.
Input files:
'../data/in/traffic.json'
'../data/in/traffic.labels'
'../data/in/sports.json'
'../data/in/sports.labels'
'../data/in/festival.json'
'../data/in/festival.labels'
'../data/in/disaster.json'
'../data/in/disaster.labels'
'../data/in/test.json'
'../data/in/test.labels'
Output files:
'../data/out/clf_train.vectors'
'../data/out/clf_train.json'
'../data/out/clf_train.labels'
'../data/out/clf_test.vectors'
'../data/out/clf_t est.json'
'../data/out/clf_test.labels'

# clusterRanking.py:
Libraries required:
* numpy
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
Description:
This module is for generating a visualization graph of a confusion matrix. It reads a list of ground truth and predictions as input and uses modules from sklearn and matplotlib to generate the graph with number and color standing for its rate. 
Input file:
'../data/out/nn_cm'
Output file:
'../data/pic/nn_cm.png'

# Cos_DBScan.py
Libraries required:
None
Description:
This program is used to do clustering on four datasets, which the input is the four predict class datas which generate during the classfication process.The result is a data file that contains datas that have been clustered to different clusters.
Input file:
'../data/out/predict_class_1.vectors'
'../data/out/predict_class_2.vectors'
'../data/out/predict_class_3.vectors'
'../data/out/predict_class_4.vectors'
Output file:
'../data/out/cluster_result_1.csv'
'../data/out/cluster_result_2.csv'
'../data/out/cluster_result_3.csv'
'../data/out/cluster_result_4.csv'

# twitter_crawler.py
Libraries required:
tweepy
Description:
Run twitter api to crawl data with specific search query. It will crawl 100 data each run.
Output file:
'../data/out/raw_tweets.json'


