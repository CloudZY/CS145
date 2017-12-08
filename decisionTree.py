from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import ast

class Model:
  def __init__(self):
    self.model = None

  def readData(self, v_path, l_path):
    vector_file = open(v_path)
    label_file = open(l_path)
    vectors = []
    ids = []
    labels = []
    words = vector_file.readline()
    for line in vector_file:
      l_vect = ast.literal_eval(line)
      vectors.append(l_vect[0:len(l_vect)-1])
      ids.append(l_vect[-1])
    for line in label_file:
      labels.append(int(line))
    return vectors, ids, labels

  def getSubset(self, vectors, labels, keys):
    index = list(keys)
    ret_vectors = [vectors[i] for i in index]
    ret_labels = [labels[i] for i in index]
    return ret_vectors, ret_labels

  def trainModel(self, v_path, l_path):
    vectors, ids, labels = self.readData(v_path, l_path)

    kf = KFold(n_splits=10, shuffle=True)
    accu = []
    models = []
    fold = 1
    for train_keys, test_keys in kf.split(range(len(vectors))):
      train_vectors, train_labels = self.getSubset(vectors, labels, train_keys)
      test_vectors, test_labels = self.getSubset(vectors, labels, test_keys)
      #clf = tree.DecisionTreeClassifier()
      clf = RandomForestClassifier(n_estimators=5)
      clf = clf.fit(train_vectors,train_labels)
      models.append(clf)

      pred = clf.predict(test_vectors)
      nfold_acc = accuracy_score(test_labels, pred)

      accu.append(nfold_acc)
      print '{} fold accuracy:{}'.format(fold, nfold_acc)
      fold += 1
    print 'average accuracy:{}'.format(np.mean(accu))

    max_acc = np.argmax(accu)
    self.model = models[max_acc]

  def testModel(self, v_path, l_path):
    vectors, ids, labels = self.readData(v_path, l_path)
    prediction = self.model.predict(vectors)
    accu = accuracy_score(labels, prediction)
    print 'test accuracy:{}'.format(accu)

    f = open('./out/decisionTree_gt_pred.txt', 'w+')
    f.write(str(labels))
    f.write('\n')
    f.write(str(prediction.tolist()))
    f.write('\n')
    f.write(str(accu))
    f.close()

    f1 = open('./out/dt_pred_class_1.vectors', 'w+')
    f2 = open('./out/dt_pred_class_2.vectors', 'w+')
    f3 = open('./out/dt_pred_class_3.vectors', 'w+')
    f4 = open('./out/dt_pred_class_4.vectors', 'w+')
    for i in range(len(vectors)):
      output = str(vectors[i]+[ids[i]])
      if prediction[i] == 1:
        f1.write(output)
        f1.write('\n')
      if prediction[i] == 2:
        f2.write(output)
        f2.write('\n')
      if prediction[i] == 3:
        f3.write(output)
        f3.write('\n')
      if prediction[i] == 4:
        f4.write(output)
        f4.write('\n')

    f1.close()
    f2.close()
    f3.close()
    f4.close()


    
if __name__ == '__main__':
  train_vector_path = 'out/clf_train.vectors'
  train_label_path = 'out/clf_train.labels'

  M = Model()
  M.trainModel(train_vector_path, train_label_path)
  M.testModel(train_vector_path, train_label_path)
