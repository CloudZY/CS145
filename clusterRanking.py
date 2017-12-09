import os
import ast
import json
import numpy
from math import log
from collections import defaultdict

# class for cluster
class cluster:
  def __init__(self, l):
    self.label = l
    self.retweet_count = 0
    self.favorite_count = 0
    self.user_followers_count = 0
    self.user_friends_count = 0
    self.ranking = 0

# read in the clustering results
def readData(filepath):
  cluster_info = defaultdict(list)
  v_file = open(filepath)

  for line in v_file:
    line = ast.literal_eval(line)
    index = line[1]
    label = line[2]
    cluster_info[label].append(index)

  return cluster_info

# read in the json data
def readJson(json_path):
  json_data = []
  with open(json_path) as jsons:
    for line in jsons:
      json_data.append(ast.literal_eval(line))
  return json_data

# calculate the ranking
def visitClusterInfo(cluster_info, json_data):
  ranking_result = []
  text_result = []
  for label in cluster_info.keys():
    clus = cluster(label)
    max_ranking = 0
    max_text = ''
    for idx in cluster_info[label]:
      #print(idx)
      for data in json_data:
        if(idx==data['index']):
          re = data['retweet_count']
          fa = data['favorite_count']
          fo = data['user_followers_count']
          fr = data['user_friends_count']
          clus.retweet_count += re
          clus.favorite_count += fa
          clus.user_followers_count += fo
          clus.user_friends_count += fr
          cur_ranking = (re+1)+(fa+1)+log(fo+1)+log(fr+1)
          clus.ranking += cur_ranking
          if(cur_ranking>max_ranking):
            max_ranking = cur_ranking
            max_text = data['text']
          break
    ranking_result.append(clus.ranking/len(cluster_info[label]))
    text_result.append(max_text)
  return (ranking_result,text_result)


# main test
# read data
json_data = readJson('out/clf_train.json')
info1 = readData('out/cluster_result_1.csv')
info2 = readData('out/cluster_result_2.csv')
info3 = readData('out/cluster_result_3.csv')
info4 = readData('out/cluster_result_4.csv')

# ranking
(ranks1,texts1) = visitClusterInfo(info1, json_data)
(ranks2,texts2) = visitClusterInfo(info2, json_data)
(ranks3,texts3) = visitClusterInfo(info3, json_data)
(ranks4,texts4) = visitClusterInfo(info4, json_data)

# output
print('**********Ranking results**********')
print('Format: (ranking metric value, original text)\n')

print('-----Top 3 results for traffic-----')
f = open('out/ranking_result_1','w')
for i in range(len(ranks1)):
  f.write(str((int(ranks1[i]), texts1[i])))
  f.write('\n')
sorted_index = numpy.argsort(ranks1)
for i in range(len(ranks1)-1,len(ranks1)-4,-1):
  print(int(ranks1[sorted_index[i]]), texts1[sorted_index[i]])
print('')
print('-----Top 3 results for sports-----')
f = open('out/ranking_result_2','w')
for i in range(len(ranks2)):
  f.write(str((int(ranks2[i]), texts2[i])))
  f.write('\n')
sorted_index = numpy.argsort(ranks2)
for i in range(len(ranks2)-1,len(ranks2)-4,-1):
  print(int(ranks2[sorted_index[i]]), texts2[sorted_index[i]])
print('')
print('-----Top 3 results for festival-----')
f = open('out/ranking_result_3','w')
for i in range(len(ranks3)):
  f.write(str((int(ranks3[i]), texts3[i])))
  f.write('\n')
sorted_index = numpy.argsort(ranks3)
for i in range(len(ranks3)-1,len(ranks3)-4,-1):
  print(int(ranks3[sorted_index[i]]), texts3[sorted_index[i]])
print('')
print('-----Top 3 results for disaster-----')
f = open('out/ranking_result_4','w')
for i in range(len(ranks4)):
  f.write(str((int(ranks4[i]), texts4[i])))
  f.write('\n')
sorted_index = numpy.argsort(ranks4)
for i in range(len(ranks4)-1,len(ranks4)-4,-1):
  print(int(ranks4[sorted_index[i]]), texts4[sorted_index[i]])
print('')
