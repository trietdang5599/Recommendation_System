import json
import os
import glob
import string
import ast
import csv
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords


import json

def read_data(file_path):
    """
       params:
           file_path: Đường dẫn đến tệp
       return:
           data: Danh sách dữ liệu đã đọc, mỗi dòng là một mẫu

    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            try:
                # Chuyển đổi từ văn bản JSON sang đối tượng Python
                raw_sample = json.loads(line)
                if 'reviewText' not in raw_sample:
                    raw_sample['reviewText'] = ''
                # Chuẩn hóa dữ liệu và thêm vào danh sách data
                data.append([raw_sample['reviewerID'],
                             raw_sample['asin'],
                             raw_sample['overall'],
                             raw_sample['reviewText']])
            except json.JSONDecodeError:
                # Bắt các lỗi khi chuyển đổi từ JSON
                # Bạn có thể xử lý các lỗi ở đây nếu cần
                pass
    return data



def softmax(x):
    """Compute the softmax of vector x.
    """
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# 加载停用词
stop_words = stopwords.words("english") + list(string.punctuation)
def word_segment(text):
    # word_seg = [i for i in word_tokenize(str(text).lower()) if i not in stop_words]
    # word_seg = text.split(" ")
    word_seg = [i for i in word_tokenize(str(text).lower())]
    return word_seg


def preprocessed(text):
    """ 3文本预处理
    """
    # 分句和词性还原， 目前只实现分句
    return text.split("\.")

def CreateAndWriteCSVByReviews(name, data):
    if not os.path.exists("feature"):
        os.makedirs("feature")
    
    filename = os.path.join("feature", name + '.csv')
    # Mở file CSV để ghi
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Ghi tiêu đề
        writer.writerow(['Key','ReviewerID', 'ItemID', 'Array'])

        # Ghi dữ liệu từ từ điển vào file
        for key, value in data.items():
            writer.writerow([key, value[0], value[1], value[2]])

    print(f'File CSV "{filename}" đã được tạo và ghi thành công.')

def CreateAndWriteCSV(name, data):
    if not os.path.exists("feature"):
        os.makedirs("feature")
    
    filename = os.path.join("feature", name + '.csv')
    # Mở file CSV để ghi
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Ghi tiêu đề
        writer.writerow(['Key', 'Array'])
        # print(data)
        # Ghi dữ liệu từ từ điển vào file
        for key, value in data.items():
            # Kiểm tra kiểu dữ liệu trước khi chuyển đổi
            if isinstance(value, np.ndarray):
                array_str = str(value.tolist())
            else:
                array_str = str(value)
            writer.writerow([key, array_str])

    print(f'File CSV "{filename}" đã được tạo và ghi thành công.')

def load_data_from_csv(file_path):
    data = {}
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            key = row['Key']
            array_str = row['Array']
            if array_str.startswith('[') and array_str.endswith(']'):
                data[key] = np.array(eval(array_str))
            else:
            # Chuyển chuỗi thành mảng Python
                array = np.array(ast.literal_eval(array_str))
                data[key] = array
    return data

def merge_csv_columns(csv_file1, id_column1, csv_file2, id_column2, value_column2, new_column):
    # Đọc dữ liệu từ file CSV thứ hai và ánh xạ ID với giá trị
    id_to_value = {}
    with open(csv_file2, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id_value = row[id_column2]
            value = row[value_column2]
            id_to_value[id_value] = value

    # Đọc dữ liệu từ file CSV đầu tiên và cập nhật dữ liệu trên đó
    updated_rows = []
    with open(csv_file1, 'r', newline='') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            id_value = row[id_column1]
            if id_value in id_to_value:
                row[new_column] = id_to_value[id_value]
            else:
                row[new_column] = ''
            updated_rows.append(row)

    # Ghi dữ liệu đã cập nhật trở lại vào file CSV đầu tiên
    with open(csv_file1, 'w', newline='') as csv_file:
        fieldnames = updated_rows[0].keys()
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(updated_rows)


if __name__ == "__main__":

    import nltk
    nltk.download('averaged_perceptron_tagger')
    # from nltk.corpus import sentiwordnet as swn
    # breakdown = swn.senti_synset('breakdown.n.03')
    # print(breakdown)