import json
import os
import glob
import string

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


if __name__ == "__main__":

    import nltk
    nltk.download('averaged_perceptron_tagger')
    # from nltk.corpus import sentiwordnet as swn
    # breakdown = swn.senti_synset('breakdown.n.03')
    # print(breakdown)