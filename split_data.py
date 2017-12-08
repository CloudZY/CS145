import ast
import numpy as np

vector_file_path = './out/clf_train.vectors'
label_file_path = './out/clf_train.labels'
vector_file = open(vector_file_path)
label_file = open(label_file_path)
vector_list, label_list, id_list = [], [], []
head = vector_file.readline()
for vector_line in vector_file.readlines():
    vector_line = ast.literal_eval(vector_line)
    label_line = label_file.readline()
    vector_data = vector_line[:-1]
    vector_id = vector_line[-1]
    label = ast.literal_eval(label_line)
    vector_list.append(vector_data)
    label_list.append(label)
    id_list.append(vector_id)

seq_array = np.arange(0, len(vector_list))
np.random.shuffle(seq_array)
# print(seq_array)

vector_train_output_path = './out/train.vectors'
label_train_output_path = './out/train.labels'
vector_test_output_path = './out/test.vectors'
label_test_output_path = './out/test.labels'

split_idx = 0.8 * len(vector_list)
vector_train_output = open(vector_train_output_path, 'w')
label_train_output = open(label_train_output_path, 'w')
vector_test_output = open(vector_test_output_path, 'w')
label_test_output = open(label_test_output_path, 'w')
vector_train_output.write(head)
vector_test_output.write(head)
for i in range(len(seq_array)):
    index = seq_array[i]
    if i <= split_idx:
        vector_list[index].append(id_list[index])
        vector_train_output.write(str(vector_list[index]) + '\n')
        label_train_output.write(str(label_list[index]) + '\n')
    else:
        vector_list[index].append(id_list[index])
        vector_test_output.write(str(vector_list[index]) + '\n')
        label_test_output.write(str(label_list[index]) + '\n')

vector_train_output.close()
label_train_output.close()
vector_test_output.close()
label_test_output.close()
