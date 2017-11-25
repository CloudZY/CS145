import ast
import sys
import string
import re
import operator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm


fields = ['created_at', 'coordinates', 'text',
'retweet_count', 'favorite_count', 'hashtags',
'user_id', 'user_mentions', 'user_name', 'user_location',
'user_description', 'user_followers_count', 'user_friends_count', 'id']

disregard = ['rt']

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

def remove_dups(data, write_filename=''):
    ids = set()
    rm_dups = []
    for tweet in data:
        if tweet['id'] not in ids:
            ids.add(tweet['id'])
            rm_dups.append(tweet)

    if write_filename != '':
        out_file = open(write_filename, 'w', encoding='utf-8')
        idx = 0
        for tweet in rm_dups:
            idx += 1
            out_file.write(str(idx))
            out_file.write(tweet['text'])
            out_file.write('\n')
        out_file.close()
    return rm_dups

def tokenize_text_from_json_data(data):
    tweets_text = []
    for tweet in data:
        tweets_text.append(tweet['text'])

    stop_words = set(stopwords.words('english'))
    translator = str.maketrans(dict.fromkeys(string.punctuation))
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    filtered_text = []
    for text in tweets_text:
        text = text.translate(translator)
        text = emoji_pattern.sub(r'', text)
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
    for words in filtered_text:
        vec = [0] * len(features)
        for w in words:
            if w in features:
                vec[features[w]] += 1 / len(words)
        feat_vector.append(vec)
    return feat_vector

def evaluate_prediction(predict, test):
    count = 0
    for i in range(len(predict)):
        if predict[i] == test[i]:
            count += 1
    print("Precision: " + str(float(count) / len(predict)))

def printHelp():
    print('-K k : keep top k (>0) number of features, keep all available features if not enough in training data.')
    print('-li f : input training data from f')
    print('-lo f : output dup_removed training data texts to f if needed')
    print('-ll f : input training labels from f')
    print('-ti f : input test data from f')
    print('-to f : output dup_removed test data texts to f if needed')
    print('-tl f : input test labels from f')
    print('--default : use default values(in source file)')

if __name__ == '__main__':

    '''**********defualt values**********'''
    K = 10

    train_in_file = 'traffic_raw_tweets.json'
    train_out_file = ''
    train_label_file = 'traffic_tweets_labels'

    test_in_file = 'processed_tweets_1.json'
    test_out_file = ''
    test_label_file = 'test_labels'
    
    if len(sys.argv) == 1:
        printHelp()
        sys.exit()
    else:
        idx = 1;
        while idx < len(sys.argv):
            if sys.argv[idx] == '-K':
                try:
                    K = int(sys.argv[idx + 1])
                except ValueError:
                    print('please input a valid number')
                    sys.exit()
            elif sys.argv[idx] == '-li':
                train_in_file = sys.argv[idx + 1]
            elif sys.argv[idx] == '-lo':
                train_out_file = sys.argv[idx + 1]
            elif sys.argv[idx] == '-ll':
                train_label_file = sys.argv[idx + 1]
            elif sys.argv[idx] == '-ti':
                test_in_file = sys.argv[idx + 1]
            elif sys.argv[idx] == '-to':
                test_out_file = sys.argv[idx + 1]
            elif sys.argv[idx] == '-tl':
                test_label_file = sys.argv[idx + 1]
            elif sys.argv[idx] != '--default':
                printHelp()
                sys.exit()
            idx += 2

    train_raw_data = read_from_file(train_in_file)
    train_raw_data = remove_dups(train_raw_data, train_out_file)
    train_data = tokenize_text_from_json_data(train_raw_data)
    features = get_features_from_train_data(train_data, K)
    feat_vector = get_feature_vectors(train_data, features)
    train_labels = read_from_file(train_label_file)

    clf = svm.SVC()
    clf.fit(feat_vector, train_labels)
    test_raw_data = read_from_file(test_in_file)
    test_raw_data = remove_dups(test_raw_data, test_out_file)
    test_data = tokenize_text_from_json_data(test_raw_data)
    test_feat_vectors = get_feature_vectors(test_data, features)
    test_labels = read_from_file(test_label_file)
    evaluate_prediction(clf.predict(test_feat_vectors), test_labels)