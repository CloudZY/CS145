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

event_types = ['festival', 'disaster', 'traffic', 'sports']
label_num = {'traffic' : 1, 'sports' : 2, 'festival' : 3, 'disaster' : 4}

disregard = ['rt', '’', 'the', 'de', 'en', 'we']

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

def tokenize_text_from_json_data(data, write_filename=''):
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

    if write_filename != '':
        out_file = open(write_filename, 'w', encoding='utf-8')
        for words in filtered_text:
            out_file.write(str(words))
            out_file.write('\n')
        out_file.close()

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
    for words in filtered_text:
        vec = [0] * (len(features) + 1)
        tr_vec = [0] * 2
        for w in words:
            if w in features:
                vec[features[w]] += 1
        tr_vec[0] = sum(vec)
        tr_vec[1] = 1. - float(tr_vec[0]) / len(words)
        vec[len(features)] = len(words)
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
    print("Precision: " + str(float(count) / len(predict)))

def write_to_file(data, filename):
    out_file = open(filename, 'w', encoding='utf-8')
    for d in data:
        out_file.write(str(d))
        out_file.write('\n')
    out_file.close()

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
    '''
    train_in_file = 'traffic_raw_tweets.json'
    train_out_file = ''
    train_label_file = 'traffic_tweets_labels'

    test_in_file = 'processed_tweets_1.json'
    test_out_file = ''
    test_label_file = 'test_labels'
    
    if len(sys.argv) == 1 or sys.argv[1] == '--help':
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
    '''

    K = 10
    feat_set = set()
    all_data = []
    all_labels = []
    feats = dict()
    clfs = dict()
    
    for event_type in event_types:
        train_in_file = './data/' + event_type + '.data'
        train_tokens_in_file = './data/' + event_type + '.tokens'
        train_label_file = './data/' + event_type + '.labels'
        train_out_file = './out/' + event_type + '.tokens'

        train_data = read_from_file(train_tokens_in_file)
        #train_raw_data = remove_dups(train_raw_data)
        #train_data = tokenize_text_from_json_data(train_raw_data, train_out_file)
        features = get_features_from_train_data(train_data, K)
        feat_vec, train_vec = get_feature_vectors(train_data, features)
        train_labels = read_from_file(train_label_file)

        for i in range(len(train_labels)):
            if train_labels[i] != 0:
                all_labels.append(train_labels[i])
                all_data.append(train_data[i])
            if train_labels[i] != label_num[event_type]:
                train_labels[i] = 0
            else:
                train_labels[i] = 1

        clfs[event_type] = svm.SVC()
        clfs[event_type].fit(train_vec, train_labels)
        feats[event_type] = features

        if feat_set is not None:
            feat_set.update(set(features))
        else:
            feat_set = set(features)
    
    all_features = get_dict_from_set(feat_set)
    all_vector = [list(all_features) + ['total']]
    feat_vec, train_vec = get_feature_vectors(all_data, all_features)
    all_vector = all_vector + feat_vec
    write_to_file(all_vector, './out/clf_train.vectors')
    write_to_file(all_labels, './out/clf_train.labels')

    test_in_file = './data/test.data'
    test_raw_data = read_from_file(test_in_file)
    test_raw_data = remove_dups(test_raw_data)
    test_data = tokenize_text_from_json_data(test_raw_data, './out/test.tokens')
    predicts = [0] * len(test_data)

    for event_type in event_types:
        feat_vec, test_vec = get_feature_vectors(test_data, feats[event_type])
        one_try = clfs[event_type].predict(test_vec)
        for i in range(len(test_data)):
            predicts[i] = predicts[i] or one_try[i]

    feat_vec, test_vec = get_feature_vectors(test_data, all_features)
    feat_vec.insert(0, list(all_features) + ['total'])
    write_to_file(feat_vec, './out/clf_test.vectors')        
    write_to_file(predicts, './out/clf_test.labels')

    truth = read_from_file('./data/test.labels')
    for i in range(len(truth)):
        if truth[i] != 0:
            truth[i] = 1
    evaluate_prediction(predicts, truth)
