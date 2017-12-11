#!/usr/bin/python
# -*- coding: utf-8 -*-
import ast
import sys
import string
import re
import io
import operator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
import numpy as np


fields = ['created_at', 'coordinates', 'text',
'retweet_count', 'favorite_count', 'hashtags',
'user_id', 'user_mentions', 'user_name', 'user_location',
'user_description', 'user_followers_count', 'user_friends_count', 'id']

event_types = ['disaster', 'festival', 'traffic', 'sports']
label_num = {'traffic' : 1, 'sports' : 2, 'festival' : 3, 'disaster' : 4}

disregard = ['rt', '’', 'the', 'de', 'en', 'we', 'los', 'el', 'ca', 'la', 'angeles', '‘', 'of', 'amp', 'l']

def process_json(json):
    if json == None:
        return None

    data = dict()
    # geolocation extraction
    if 'coordinates' in json and json['coordinates'] != None:
        if 'coordinates' in json['coordinates']:
            coords = json['coordinates']['coordinates']
            if coords != None:
                data['coordinates'] = (coords[0], coords[1])
    elif 'geo' in json and json['geo'] != None:
        if 'type' in json['geo'] and json['geo']['type'] == 'Point' and 'coordinates' in json['geo']:
            coords = json['geo']['coordinates']
            if coords != None:
                data['coordinates'] = (coords[0], coords[1])

    # entities extraction
    if 'entities' in json and json['entities'] != None:
        entities = json['entities']
        if 'hashtags' in entities:
            hashtags = entities['hashtags']
            if hashtags != None:
                hts = []
                for hashtag in hashtags:
                    if 'text' in hashtag:
                        hts.append(hashtag['text'])
                data['hashtags'] = hts

        if 'user_mentions' in entities:
            user_mentions = entities['user_mentions']
            if user_mentions != None:
                ums = []
                for um in user_mentions:
                    if 'id' in um:
                        ums.append(um['id'])
                data['user_mentions'] = ums

    # user info extraction
    if 'user' in json and json['user'] != None:
        user = json['user']
        for keyword in ['id', 'name', 'location', 'description', 'followers_count', 'friends_count']:
            if keyword in user:
                data['user_' + keyword] = user[keyword]

    # other info extraction
    for field in fields:
        if not field in data and field in json:
            data[field] = json[field]

    return data

def preprocess_data(json_data):
    processed_data = []
    for json in json_data:
        j = process_json(json)
        if j is not None:
            processed_data.append(j)
        if 'retweeted_status' in json:
            j = process_json(json['retweeted_status'])
            if j is not None:
                processed_data.append(j)
    return processed_data

def print_json(tweet):
    print('-----------------------------------')
    for field in fields:
        print('...' + field + '...')
        print(tweet[field])

def read_from_file(filename):
    in_file = open(filename, 'r')
    data = []
    for line in in_file:
        data.append(ast.literal_eval(line))
    in_file.close()
    return data

def remove_dups(data):
    ids = set()
    rm_dups = []
    for tweet in data:
        if tweet['id'] not in ids:
            ids.add(tweet['id'])
            rm_dups.append(tweet)

    return rm_dups

def tokenize_text_from_json_data(data):
    tweets_text = []
    for tweet in data:
        tweets_text.append(tweet['text'])

    stop_words = set(stopwords.words('english'))
    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)

    filtered_text = []
    for text in tweets_text:
        for p in string.punctuation:
            text = text.replace(p, '')
        text = emoji_pattern.sub(r'', text)
        text = re.sub('[…\“\”\—\’]', '', text)
        text = re.sub(r'http\S*', '', text)
        word_tokens = word_tokenize(text)
        words = [w.lower() for w in word_tokens if not w in stop_words]
        filtered_text.append(words)

    return filtered_text

def get_features_from_train_data(data, K=10):
    wc = dict()
    for words in data:
        for word in words:
            if word not in wc:
                wc[word] = 1
            else:
                wc[word] = wc[word] + 1

    sorted_wc = sorted(wc.items(), key=operator.itemgetter(1), reverse=True)

    features = dict()
    counter = 0
    for key, value in sorted_wc:
        if counter == K:
            break
        if key not in disregard:
            features[key] = counter
            counter += 1
    return features

def get_feature_vectors(filtered_text, features):
    feat_vector = []
    train_vec = []
    for i in range(len(filtered_text)):
        words = filtered_text[i]
        vec = [0] * (len(features) + 1)
        tr_vec = [0] * 2
        for w in words:
            if w in features:
                vec[features[w]] += 1
        tr_vec[0] = sum(vec)
        if len(words) == 0:
            tr_vec[1] = 1.
        else:
            tr_vec[1] = 1. - float(tr_vec[0]) / len(words)
        vec[len(features)] = i + 1
        train_vec.append(tr_vec)
        feat_vector.append(vec)

    return feat_vector, train_vec

def get_dict_from_set(fset):
    counter = 0
    d = dict()
    for e in fset:
        d[e] = counter
        counter += 1
    return d

def evaluate_prediction(predict, test):
    count = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            count += 1
    return float(count) / len(predict)

def write_to_file(data, filename):
    out_file = open(filename, 'w')
    for d in data:
        out_file.write(str(d))
        out_file.write('\n')
    out_file.close()

def get_data_files_from_raw_json(in_file, out_token, out_json):
    train_raw_data = read_from_file(in_file)
    train_raw_data = preprocess_data(train_raw_data)
    train_raw_data = remove_dups(train_raw_data)
    for i in range(len(train_raw_data)):
        train_raw_data[i]['index'] = i + 1
    write_to_file(train_raw_data, out_json)
    tokens = tokenize_text_from_json_data(train_raw_data)
    for i in range(len(tokens)):
        tokens[i].append(i + 1)
    write_to_file(tokens, out_token)

def get_stats(jsons):
    users = set()
    followers = 0
    friends = 0
    start = None
    end = None
    for json in jsons:
        users.add(json['user_id'])
        followers += json['user_followers_count']
        friends += json['user_friends_count']
        if start is None:
            start = json['created_at']
        elif json['created_at'] < start:
            start = json['created_at']
        if end is None:
            end = json['created_at']
        elif json['created_at'] > end:
            end = json['created_at']
    return users, followers, friends, start, end

if __name__ == '__main__':

    run_cv = False
    if len(sys.argv) == 2 and sys.argv[1] == '-cv':
        run_cv = True

    K = 10
    feat_set = set()
    non_noise_data_tokens = []
    non_noise_json = []
    non_noise_data_labels = []
    train_feature_dict = dict()
    train_clfs = dict()

    '''**********Model***********'''
    for event_type in event_types:
        train_json_in_file = '../data/in/' + event_type + '.json'
        #train_tokens_in_file = '../data/in/' + event_type + '.tokens'
        train_label_in_file = '../data/in/' + event_type + '.labels'
        #train_tokens_out_file = '../data/out/' + event_type + '.tokens'

        # data preparation
        train_json_data = remove_dups(preprocess_data(read_from_file(train_json_in_file)))
        train_tokens = tokenize_text_from_json_data(train_json_data)
        train_feature = get_features_from_train_data(train_tokens, K)
        train_feature_vector, train_vector = get_feature_vectors(train_tokens, train_feature)
        train_labels = read_from_file(train_label_in_file)

        # mark non event_type labels and event_type labels
        for i in range(len(train_labels)):
            if train_labels[i] != 0:
                non_noise_data_labels.append(train_labels[i])
                non_noise_data_tokens.append(train_tokens[i])
                non_noise_json.append(train_json_data[i])
            if train_labels[i] != label_num[event_type]:
                train_labels[i] = 0
            else:
                train_labels[i] = 1

        # SVM for this event_type
        train_clfs[event_type] = SVC().fit(train_vector, train_labels)
        train_feature_dict[event_type] = train_feature

        # integrate features
        if feat_set is not None:
            feat_set.update(set(train_feature))
        else:
            feat_set = set(train_feature)


        # cross validation
        if run_cv:
            train_vector = np.array(train_vector)
            train_labels = np.array(train_labels)
            test_accuracy = 0
            cmat = np.zeros((2,2), dtype='float')
            cv = cross_validation.KFold(len(train_vector), n_folds = 10)
            for traincv, testcv in cv:
                train_x = train_vector[traincv]
                train_y = train_labels[traincv]
                test_x = train_vector[testcv]
                test_y = train_labels[testcv]
                test_clf = SVC().fit(train_x, train_y)
                predicts = test_clf.predict(test_x)
                test_accuracy += evaluate_prediction(predicts, test_y) / 10
                cmat += np.array(confusion_matrix(test_y, predicts), dtype='float') / 10          
            print(event_type + " Accuracy: ", test_accuracy)
            print(event_type + " Confusion Matrix: ", cmat)
    
    # add index to non noise json data
    for i in range(len(non_noise_json)):
        non_noise_json[i]['index'] = i + 1

    # integrate features and get output data vectors for training classifier
    all_features = get_dict_from_set(feat_set)
    non_noise_feature_vector = [list(all_features) + ['id']]
    feature_vector, non_noise_train_vector = get_feature_vectors(non_noise_data_tokens, all_features)
    non_noise_feature_vector = non_noise_feature_vector + feature_vector
    write_to_file(non_noise_feature_vector, '../data/out/clf_train.vectors')
    write_to_file(non_noise_data_labels, '../data/out/clf_train.labels')
    write_to_file(non_noise_json, '../data/out/clf_train.json')

    # read and prepare test data
    test_json_in_file = '../data/in/test.json'
    test_json_data = read_from_file(test_json_in_file)
    test_tokens = tokenize_text_from_json_data(test_json_data)
    predicts = [0] * len(test_tokens)

    # check if an tweet is an event
    for event_type in event_types:
        test_feature_vector, test_vector = get_feature_vectors(test_tokens, train_feature_dict[event_type])
        one_try = train_clfs[event_type].predict(test_vector)
        for i in range(len(test_tokens)):
            predicts[i] = predicts[i] or one_try[i]

    # evaluate test data accuracy
    test_labels = read_from_file('../data/in/test.labels')
    truth = list(test_labels)
    for i in range(len(truth)):
        if truth[i] != 0:
            truth[i] = 1
    print("Accuracy: ", evaluate_prediction(predicts, truth))

    filtered_test_tokens = []
    filtered_test_labels = []
    filtered_test_json = []
    for i in range(len(test_tokens)):
        if predicts[i] == 1:
            filtered_test_tokens.append(test_tokens[i])
            filtered_test_labels.append(test_labels[i])
            filtered_test_json.append(test_json_data[i])

    # generate test data vector
    test_feature_vector, test_vector = get_feature_vectors(filtered_test_tokens, all_features)
    test_feature_vector.insert(0, list(all_features) + ['id'])
    for i in range(len(filtered_test_json)):
        filtered_test_json[i]['index'] = i + 1
    write_to_file(test_feature_vector, '../data/out/clf_test.vectors')        
    write_to_file(filtered_test_json, '../data/out/clf_test.json')
    write_to_file(filtered_test_labels, '../data/out/clf_test.labels')
    
   