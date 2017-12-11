from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import ast

class Model:
  def __init__(self):
    self.dt_model = None
    self.rf_model = None

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
    
    dt_models = []
    dt_accu = []

    rf_models = []
    rf_accu = []
    for train_keys, test_keys in kf.split(range(len(vectors))):
      train_vectors, train_labels = self.getSubset(vectors, labels, train_keys)
      test_vectors, test_labels = self.getSubset(vectors, labels, test_keys)
      
      dt_clf = tree.DecisionTreeClassifier()
      dt_clf = dt_clf.fit(train_vectors, train_labels)
      dt_models.append(dt_clf)
      dt_pred = dt_clf.predict(test_vectors)
      dt_nfold_acc = accuracy_score(test_labels, dt_pred)
      dt_accu.append(dt_nfold_acc)

      rf_clf = RandomForestClassifier(n_estimators=10)
      rf_clf = rf_clf.fit(train_vectors,train_labels)
      rf_models.append(rf_clf)
      rf_pred = rf_clf.predict(test_vectors)
      rf_nfold_acc = accuracy_score(test_labels, rf_pred)
      rf_accu.append(rf_nfold_acc)

    print 'decision tree 10-fold average accuracy:{}'.format(np.mean(dt_accu))
    print 'random forest 10-fold average accuracy:{}'.format(np.mean(rf_accu))

    dt_max_acc = np.argmax(dt_accu)
    rf_max_acc = np.argmax(rf_accu)

    self.dt_model = dt_models[dt_max_acc]
    self.rf_model = rf_models[rf_max_acc]

  def testModel(self, v_path, l_path, tv_path, tl_path):
    t_vectors, t_ids, t_labels = self.readData(tv_path, tl_path)
    t_dt_prediction = self.dt_model.predict(t_vectors)
    t_rf_prediction = self.rf_model.predict(t_vectors)
    t_dt_accu = accuracy_score(t_labels, t_dt_prediction)
    t_rf_accu = accuracy_score(t_labels, t_rf_prediction)

    vectors, ids, labels = self.readData(v_path, l_path)
    dt_prediction = self.dt_model.predict(vectors)
    rf_prediction = self.rf_model.predict(vectors)
    dt_accu = accuracy_score(labels, dt_prediction)
    rf_accu = accuracy_score(labels, rf_prediction)

    # uncomment the following two lines if you want accuracy on test dataset
    # print 'decision tree test accuracy:{}'.format(t_dt_accu)
    # print 'random forest test accuracy:{}'.format(t_rf_accu)
    print 'decision tree whole dataset accuracy:{}'.format(dt_accu)
    print 'random forest whole dataset accuracy:{}'.format(rf_accu)

    f = open('../data/out/dt_cm', 'w+')
    f.write(str(labels))
    f.write('\n')
    f.write(str(dt_prediction.tolist()))
    f.write('\n')
    f.write(str(dt_accu))
    f.close()

    f = open('../data/out/rf_cm', 'w+')
    f.write(str(labels))
    f.write('\n')
    f.write(str(rf_prediction.tolist()))
    f.write('\n')
    f.write(str(rf_accu))
    f.close()

    f1 = open('../data/out/dt_predict_class_1.vectors', 'w')
    f2 = open('../data/out/dt_predict_class_2.vectors', 'w')
    f3 = open('../data/out/dt_predict_class_3.vectors', 'w')
    f4 = open('../data/out/dt_predict_class_4.vectors', 'w')
    for i in range(len(vectors)):
      output = generate_predict_output(vectors[i], ids[i])+'\n'
      if dt_prediction[i] == 1:
        f1.write(output)
      if dt_prediction[i] == 2:
        f2.write(output)
      if dt_prediction[i] == 3:
        f3.write(output)
      if dt_prediction[i] == 4:
        f4.write(output)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

    f1 = open('../data/out/rf_predict_class_1.vectors', 'w')
    f2 = open('../data/out/rf_predict_class_2.vectors', 'w')
    f3 = open('../data/out/rf_predict_class_3.vectors', 'w')
    f4 = open('../data/out/rf_predict_class_4.vectors', 'w')
    for i in range(len(vectors)):
      output = generate_predict_output(vectors[i], ids[i])+'\n'
      if rf_prediction[i] == 1:
        f1.write(output)
      if rf_prediction[i] == 2:
        f2.write(output)
      if rf_prediction[i] == 3:
        f3.write(output)
      if rf_prediction[i] == 4:
        f4.write(output)
    f1.close()
    f2.close()
    f3.close()
    f4.close()

def generate_predict_output(vector, id):
  tmp = [i for i in vector]
  tmp.append(id)
  data = str(tmp)[1:-1].replace(', ', '\t')
  return data

def runModel(train_vector_path, train_label_path, test_vector_path, test_label_path):
  M = Model()
  M.trainModel(train_vector_path, train_label_path)
  M.testModel(train_vector_path, train_label_path, test_vector_path, test_label_path)
    
if __name__ == '__main__':
  train_vector_path = '../data/out/clf_train.vectors'
  train_label_path = '../data/out/clf_train.labels'
  test_vector_path = '../data/out/clf_test.vectors'
  test_label_path = '../data/out/clf_test.labels'

  runModel(train_vector_path, train_label_path, test_vector_path, test_label_path)
