# CS145

# in/[disaster, traffic, sports, festival].json:
input json training data file

# in/[disaster, traffic, sports, festival].json:
input training labels

# in/test.*:
sample input test data

# out/clf_train.json:
Json data file of non noise training data

# out/clf_train.vectors:
feature vectors of non noise training data

# out/clf_train.labels:
ground truth labels of non noise training data

# out/clf_test.*:
sample processed test data

# out/dt[rf]_cm:
These are the files generated along with the classification of decision tree(dt) and random forest(rf). They are used to draw the confusion matrix diagram. The first line is the groun truth list and the second line is the prediction list. The third line may contains the accuracy of test and train set.

# out/dt[rf]_predict_class_[1,2,3,4].vectors:
These files are the classification results of decision tree(dt) and random forest(rf). The number 1,2,3,4 are correponding to different classes, where 1 stands for traffic, 2 stands for sports, 3 stands for disaster, 4 stands for festival. Each line is a list. The last element of the list is the original id of this tweet and the rest part of it is the feature vector of this tweet.

# out/nn_cm:
These are the files generated along with the classification of neural network. They are used to draw the confusion matrix diagram. The first line is the groun truth list and the second line is the prediction list. The third line may contains the accuracy of test and train set.

# out/nn_predict_class_[1,2,3,4].vectors:
These files are the classification results of neural network. The number 1,2,3,4 are corresponding to different classes, where 1 stands for traffic, 2 stands for sports, 3 stands for disaster, 4 stands for festival. Each line is a list. The last element of the list is the original id of this tweet and the rest part of it is the feature vector of this tweet.

# pic/dt_cm.png, nn_cm.png, rf_cm.png
These pictures are visualization grpahs of the confusion matrixs according to each classifier's results. The darker the color is, the high rate it is.

# out/cluster_result_[1,2,3,4].csv:
These files are the clustering result of dbscan clusterings. The number 1,2,3,4 are corresponding to four different classes, where 1 stands for traffic, 2 stands for sports, 3 stands for disaster, 4 stands for festival. For each line of the data file, there first part is a list of key word bags that one tweets contains, the second part is the tweet's id, the third part is the cluster class id.

# out/raw_tweets.json
raw json tweets data crawled using twitter api

# out/cos_DBSCAN.csv
running results of DBCSAN clustering