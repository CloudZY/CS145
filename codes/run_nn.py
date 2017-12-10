import ast
import torch
from torch.autograd import Variable
from codes.neural_network import TwoLayerNet
import numpy as np
from sklearn import cross_validation

def read_vector_label(vector_file_path, label_file_path):
    vector_file = open(vector_file_path)
    label_file = open(label_file_path)
    vector_list, label_list, id_list = [], [], []
    vector_file.readline()
    for vector_line in vector_file.readlines():
        vector_line = ast.literal_eval(vector_line)
        label_line = label_file.readline()
        vector_data = vector_line[:-1]
        vector_id = vector_line[-1]
        label = ast.literal_eval(label_line)-1
        vector_list.append(vector_data)
        label_list.append(label)
        id_list.append(vector_id)
    seq_array = np.arange(0, len(vector_list))
    np.random.shuffle(seq_array)
    vector_list = [vector_list[index] for index in seq_array]
    label_list = [label_list[index] for index in seq_array]
    id_list = [id_list[index] for index in seq_array]
    return vector_list, label_list, id_list

def generate_predict_output(vector, id):
    tmp = vector
    tmp.append(id)
    data = str(tmp)[1:-1].replace(', ', '\t')
    return data

# def main():
#     vector_file_path = '../data/out/train.vectors'
#     label_file_path = '../data/out/train.labels'
#     D_in, H, D_out, epoch_num, k_fold = 39, 200, 4, 1500, 10
#
#     vector_list, label_list, id_list = read_vector_label(vector_file_path, label_file_path)
#     cv = cross_validation.KFold(len(vector_list), n_folds=5)
#
#
#     model = TwoLayerNet(D_in, H, D_out)
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
#
#     vector_file_test_path = './out/test.vectors'
#     label_file_test_path = './out/test.labels'
#
#     vector_list_test, label_list_test, id_list_test = read_vector_label(vector_file_test_path, label_file_test_path)
#
#     test_x = Variable(torch.FloatTensor(vector_list_test))
#     test_y = Variable(torch.LongTensor(label_list_test))
#
#     final_label_result = []
#     final_train_result = []
#     train_acc = 0.0
#     best_acc = 0.0
#
#     for t in range(epoch_num):
#         # Forward pass: Compute predicted y by passing x to the model
#         x = Variable(torch.FloatTensor(vector_list))
#         y = Variable(torch.LongTensor(label_list))
#         y_pred = model(x)
#         # Compute accuracy
#         count = 0
#         value, pred_label = torch.max(y_pred, 1)
#         for index in range(len(y)):
#             # print(pred_label[index].data[0], " ", y[index].data[0])
#             if pred_label[index].data[0] == y[index].data[0]:
#                 count += 1
#         print("epoch ", t, " : ", count / len(y))
#
#         # Compute and print loss
#         loss = criterion(y_pred, y)
#         # print(t, loss.data[0])
#
#         # Zero gradients, perform a backward pass, and update the weights.
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         correct_count = 0
#         y_pred = model(test_x)
#         value, test_pred_label = torch.max(y_pred, 1)
#         for index in range(len(test_y)):
#             if test_pred_label[index].data[0] == test_y[index].data[0]:
#                 correct_count += 1
#         print("test accuracy: ", correct_count / len(test_y))
#         if (correct_count / len(test_y) > best_acc):
#             final_label_result = test_pred_label
#             final_train_result = pred_label
#             train_acc = count / len(y)
#             best_acc = correct_count / len(test_y)
#
#     predict_file_1 = open('./out/predict_class_1.vectors', 'w')
#     predict_file_2 = open('./out/predict_class_2.vectors', 'w')
#     predict_file_3 = open('./out/predict_class_3.vectors', 'w')
#     predict_file_4 = open('./out/predict_class_4.vectors', 'w')
#     for index in range(len(test_y)):
#         if final_label_result[index].data[0] == 0:
#             predict_file_1.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')
#         elif final_label_result[index].data[0] == 1:
#             predict_file_2.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')
#         elif final_label_result[index].data[0] == 2:
#             predict_file_3.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')
#         elif final_label_result[index].data[0] == 3:
#             predict_file_4.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')
#
#     for index in range(len(label_list)):
#         if final_train_result[index].data[0] == 0:
#             predict_file_1.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')
#         elif final_train_result[index].data[0] == 1:
#             predict_file_2.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')
#         elif final_train_result[index].data[0] == 2:
#             predict_file_3.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')
#         elif final_train_result[index].data[0] == 3:
#             predict_file_4.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')
#
#     print(len(vector_list))
#     print(len(label_list))
#     final_train_result = [final_train_result[i].data[0] for i in range(len(label_list))]
#     final_label_result = [final_label_result[i].data[0] for i in range(len(label_list_test))]
#     label_list = label_list + label_list_test
#     final_train_result = final_train_result + final_label_result
#     print(len(final_train_result))
#     print(len(label_list))
#     cm_file = open('./out/nn_cm', 'w')
#     cm_file.write(str(label_list) + '\n')
#     cm_file.write(str(final_train_result) + '\n')
#     cm_file.write(str(train_acc) + ' ' + str(best_acc))
#     cm_file.close()

def main():
    # vector_file_path = '../data/out/train.vectors'
    # label_file_path = '../data/out/train.labels'
    D_in, H, D_out, epoch_num, k_fold = 39, 200, 4, 1500, 5

    vector_list, label_list, id_list = read_vector_label('../data/out/clf_train.vectors', '../data/out/clf_train.labels')
    cv = cross_validation.KFold(len(vector_list), n_folds=k_fold)

    best_test_result = []
    best_train_result = []
    best_test_truth = []
    best_train_truth = []

    best_train_vector = []
    best_train_id = []
    best_test_vector = []
    best_test_id = []

    best_train_acc = 0.0
    best_test_acc = 0.0
    train_acc_avg = 0.0
    test_acc_avg = 0.0

    fold_count = 1

    for traincv, testcv in cv:
        print("------------------------------------")
        print('Start Fold ', fold_count, ' validation')
        fold_count += 1
        train_vector = np.array(vector_list)[traincv].tolist()
        train_label = np.array(label_list)[traincv].tolist()
        train_id = np.array(id_list)[traincv].tolist()
        test_vector = np.array(vector_list)[testcv].tolist()
        test_label = np.array(label_list)[testcv].tolist()
        test_id = np.array(id_list)[testcv].tolist()
        model_test_truth = test_label
        model_train_truth = train_label

        model = TwoLayerNet(D_in, H, D_out)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

        train_x = Variable(torch.FloatTensor(train_vector))
        train_y = Variable(torch.LongTensor(train_label))
        test_x = Variable(torch.FloatTensor(test_vector))
        test_y = Variable(torch.LongTensor(test_label))

        model_test_result = []
        model_train_result = []
        model_train_acc = 0.0
        model_test_acc = 0.0

        for t in range(epoch_num):
            # Forward pass: Compute predicted y by passing x to the model
            train_y_pred = model(train_x)
            # Compute accuracy
            count = 0
            value, pred_label = torch.max(train_y_pred, 1)
            for index in range(len(train_y_pred)):
                if pred_label[index].data[0] == train_y[index].data[0]:
                    count += 1
            print("epoch ", t, " : ", count / len(train_y_pred))

            # Compute and print loss
            loss = criterion(train_y_pred, train_y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct_count = 0
            test_y_pred = model(test_x)
            value, test_pred_label = torch.max(test_y_pred, 1)
            for index in range(len(test_y_pred)):
                if test_pred_label[index].data[0] == test_y[index].data[0]:
                    correct_count += 1
            print("test accuracy: ", correct_count / len(test_y))
            if (correct_count / len(test_y) > model_test_acc):
                model_test_result = test_pred_label
                model_train_result = pred_label
                model_train_acc = count / len(train_y)
                model_test_acc = correct_count / len(test_y)

        if model_test_acc > best_test_acc:
            best_test_result = model_test_result
            best_train_result = model_train_result
            best_train_acc = model_train_acc
            best_test_acc = model_test_acc
            best_test_truth = model_test_truth
            best_train_truth = model_train_truth
            best_train_vector = train_vector
            best_train_id = train_id
            best_test_vector = test_vector
            best_test_id = test_id

        train_acc_avg += model_train_acc
        test_acc_avg += model_test_acc
    train_acc_avg /= k_fold
    test_acc_avg /= k_fold
    total_acc_avg = test_acc_avg * (1.0 / k_fold) + train_acc_avg * (1 - 1.0 / k_fold)
    print("------------------------------------")
    print('Average accuracy for Neural Network: ', total_acc_avg)

    predict_file_1 = open('../data/out/nn_predict_class_1.vectors', 'w')
    predict_file_2 = open('../data/out/nn_predict_class_2.vectors', 'w')
    predict_file_3 = open('../data/out/nn_predict_class_3.vectors', 'w')
    predict_file_4 = open('../data/out/nn_predict_class_4.vectors', 'w')
    for index in range(len(best_test_truth)):
        if best_test_result[index].data[0] == 0:
            predict_file_1.write(generate_predict_output(best_test_vector[index], best_test_id[index]) + '\n')
        elif best_test_result[index].data[0] == 1:
            predict_file_2.write(generate_predict_output(best_test_vector[index], best_test_id[index]) + '\n')
        elif best_test_result[index].data[0] == 2:
            predict_file_3.write(generate_predict_output(best_test_vector[index], best_test_id[index]) + '\n')
        elif best_test_result[index].data[0] == 3:
            predict_file_4.write(generate_predict_output(best_test_vector[index], best_test_id[index]) + '\n')

    for index in range(len(best_train_truth)):
        if best_train_result[index].data[0] == 0:
            predict_file_1.write(generate_predict_output(best_train_vector[index], best_train_id[index]) + '\n')
        elif best_train_result[index].data[0] == 1:
            predict_file_2.write(generate_predict_output(best_train_vector[index], best_train_id[index]) + '\n')
        elif best_train_result[index].data[0] == 2:
            predict_file_3.write(generate_predict_output(best_train_vector[index], best_train_id[index]) + '\n')
        elif best_train_result[index].data[0] == 3:
            predict_file_4.write(generate_predict_output(best_train_vector[index], best_train_id[index]) + '\n')

    best_train_result = [best_train_result[i].data[0] for i in range(len(best_train_result))]
    best_test_result = [best_test_result[i].data[0] for i in range(len(best_test_result))]
    final_truth = best_train_truth + best_test_truth
    final_result = best_train_result + best_test_result
    cm_file = open('../data/out/nn_cm', 'w')
    cm_file.write(str(final_truth) + '\n')
    cm_file.write(str(final_result) + '\n')
    cm_file.write(str(best_train_acc) + ' ' + str(best_test_acc))
    cm_file.close()

if __name__ == '__main__':
    main()
