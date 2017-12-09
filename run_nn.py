import ast
import torch
from torch.autograd import Variable
from neural_network import TwoLayerNet
import numpy as np

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
    return vector_list, label_list, id_list

def generate_folds(vector_list, label_list, id_list, K):
    fold_size = int(len(vector_list) / K) + 1
    print('data: ', len(vector_list))
    print('fold size: ', fold_size)
    idx_list = []
    for i in np.arange(0, len(vector_list), fold_size):
        idx_list.append(np.arange(i, min(i+fold_size, len(vector_list))))
    folds = []
    for idx in idx_list:
        print([vector_list[i] for i in idx])
        folds.append([[vector_list[i] for i in idx], [label_list[i] for i in idx], [id_list[i] for i in idx]])
    return folds

def generate_predict_output(vector, id):
    tmp = vector
    tmp.append(id)
    data = str(tmp)[1:-1].replace(', ', '\t')
    return data

def main():
    vector_file_path = './out/train.vectors'
    label_file_path = './out/train.labels'
    D_in, H, D_out, epoch_num, k_fold = 39, 200, 4, 1500, 10

    vector_list, label_list, id_list = read_vector_label(vector_file_path, label_file_path)

    model = TwoLayerNet(D_in, H, D_out)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    vector_file_test_path = './out/test.vectors'
    label_file_test_path = './out/test.labels'

    vector_list_test, label_list_test, id_list_test = read_vector_label(vector_file_test_path, label_file_test_path)

    test_x = Variable(torch.FloatTensor(vector_list_test))
    test_y = Variable(torch.LongTensor(label_list_test))

    final_label_result = []
    final_train_result = []
    train_acc = 0.0
    best_acc = 0.0

    for t in range(epoch_num):
        # Forward pass: Compute predicted y by passing x to the model
        x = Variable(torch.FloatTensor(vector_list))
        y = Variable(torch.LongTensor(label_list))
        y_pred = model(x)
        # Compute accuracy
        count = 0
        value, pred_label = torch.max(y_pred, 1)
        for index in range(len(y)):
            # print(pred_label[index].data[0], " ", y[index].data[0])
            if pred_label[index].data[0] == y[index].data[0]:
                count += 1
        print("epoch ", t, " : ", count / len(y))

        # Compute and print loss
        loss = criterion(y_pred, y)
        # print(t, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_count = 0
        y_pred = model(test_x)
        value, test_pred_label = torch.max(y_pred, 1)
        for index in range(len(test_y)):
            if test_pred_label[index].data[0] == test_y[index].data[0]:
                correct_count += 1
        print("test accuracy: ", correct_count / len(test_y))
        if (correct_count / len(test_y) > best_acc):
            final_label_result = test_pred_label
            final_train_result = pred_label
            train_acc = count / len(y)
            best_acc = correct_count / len(test_y)

    predict_file_1 = open('./out/predict_class_1.vectors', 'w')
    predict_file_2 = open('./out/predict_class_2.vectors', 'w')
    predict_file_3 = open('./out/predict_class_3.vectors', 'w')
    predict_file_4 = open('./out/predict_class_4.vectors', 'w')
    for index in range(len(test_y)):
        if final_label_result[index].data[0] == 0:
            predict_file_1.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')
        elif final_label_result[index].data[0] == 1:
            predict_file_2.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')
        elif final_label_result[index].data[0] == 2:
            predict_file_3.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')
        elif final_label_result[index].data[0] == 3:
            predict_file_4.write(generate_predict_output(vector_list_test[index], id_list_test[index]) + '\n')

    for index in range(len(label_list)):
        if final_train_result[index].data[0] == 0:
            predict_file_1.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')
        elif final_train_result[index].data[0] == 1:
            predict_file_2.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')
        elif final_train_result[index].data[0] == 2:
            predict_file_3.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')
        elif final_train_result[index].data[0] == 3:
            predict_file_4.write(generate_predict_output(vector_list[index], id_list[index]) + '\n')

    print(len(vector_list))
    print(len(label_list))
    final_train_result = [final_train_result[i].data[0] for i in range(len(label_list))]
    final_label_result = [final_label_result[i].data[0] for i in range(len(label_list_test))]
    label_list = label_list + label_list_test
    final_train_result = final_train_result + final_label_result
    print(len(final_train_result))
    print(len(label_list))
    cm_file = open('./out/nn_cm', 'w')
    cm_file.write(str(label_list) + '\n')
    cm_file.write(str(final_train_result) + '\n')
    cm_file.write(str(train_acc) + ' ' + str(best_acc))
    cm_file.close()

if __name__ == '__main__':
    main()