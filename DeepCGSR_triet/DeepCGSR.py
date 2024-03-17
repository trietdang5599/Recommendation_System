import numpy as np 
import ast
import csv
import os
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import sentiwordnet as swn
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.parse.stanford import StanfordDependencyParser
import pandas as pd
from gensim.models import word2vec, Word2Vec
from sklearn.metrics import roc_auc_score
import tqdm
from utils import read_data, softmax, word_segment, preprocessed, sigmoid
import torch
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')

#region Fine-gain
# ============== fine-gain ================
#step1: LDA model

def get_lda_mdoel(split_data, num_topics, num_words):
    """ LDA模型训练词表构建主题单词矩阵获取
    """

    # 构建词表
    dictionary = corpora.Dictionary(split_data)
    corpus = [dictionary.doc2bow(text) for text in split_data]

    # LDA模型训练
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)

    # 主题单词矩阵
    topic_to_words = []
    for i in range(num_topics):
        cur_topic_words = [ele[0] for ele in model.show_topic(i, num_words)]
        topic_to_words.append(cur_topic_words)
    return model, dictionary, topic_to_words

# step2: nltk 依存句法分析

class DependencyParser():
    def __init__(self, model_path, parser_path):
        # self.model = CoreNLPDependencyParser(url='http://localhost:9000')
        self.model = StanfordDependencyParser(path_to_jar=parser_path, path_to_models_jar=model_path)

    def raw_parse(self, text):
        parse_result = self.model.raw_parse(text)
        result = [list(parse.triples()) for parse in parse_result]
        return result[0]
    
# step3: sentimentWordNet 词云获取单词情感
#
# n = ['NN','NNP','NNPS','NNS','UH']
# v = ['VB','VBD','VBG','VBN','VBP','VBZ']
# a = ['JJ','JJR','JJS']
# r = ['RB','RBR','RBS','RP','WRB']

def get_word_sentiment_score(word):
    m = list(swn.senti_synsets(word, "n"))
    s = 0
    for j in range(len(m)):
        # print(m[j])
        s += (m[j].pos_score() - m[j].neg_score())
    return s

#-----
def get_topic_sentiment_metrix(text, dictionary, lda_model, topic_word_metrix, dependency_parser, topic_nums=50):

    """获取主题-情感矩阵
    """
    text_p = word_segment(text)
    doc_bow = dictionary.doc2bow(text_p)  # 文档转换成bow
    doc_lda = lda_model[doc_bow]  # [(12, 0.042477883), (13, 0.36870235), (16, 0.35455772), (37, 0.20635633)]

    # 初始化主题矩阵
    topci_sentiment_m = np.zeros(topic_nums)

    # 获取依存句法分析结果
    sentences = preprocessed(text)
    dep_parser_result_p = []
    for i in sentences:
        # 依存句法分析
        # print(i)
        dep_parser_result = dependency_parser.raw_parse(i)
        # print(dep_parser_result)
        for j in dep_parser_result:
            dep_parser_result_p.append([j[0][0], j[2][0]])
    #     print(dep_parser_result_p)
    # print(doc_lda)
    for topic_id, _ in doc_lda:
        # 获取当前主题的特征词
        cur_topic_words = topic_word_metrix[topic_id]
        cur_topic_sentiment = 0
        cur_topci_senti_word = []

        # 根据特征词获取情感词
        # print("当前句子", word_segment(text))
        for word in word_segment(text):
            # 获取当前文本出现的特征词
            if word in cur_topic_words:
                cur_topci_senti_word.append(word)
                # 根据依存关系， 获得依存词。 并计算主题情感
                for p in dep_parser_result_p:
                    if p[0] == word:
                        # 将依存词的情感加入主题
                        cur_topci_senti_word.append(p[1])
                    if p[1] == word:
                        cur_topci_senti_word.append(p[0])

        for senti_word in cur_topci_senti_word:
            # cur_topic_sentiment += word_to_senti.get(senti_word, 0)
            cur_topic_sentiment += get_word_sentiment_score(senti_word)
        # print("cur_topci_senti_word", cur_topci_senti_word)
        # 主题情感取值范围[-5, 5]
        if cur_topic_sentiment > 5:
            cur_topic_sentiment = 5
        elif cur_topic_sentiment < -5:
            cur_topic_sentiment = -5

        topci_sentiment_m[topic_id] = cur_topic_sentiment
    return topci_sentiment_m

#=============== End fine-gain functions ===================
#endregion

#region Coarse-gain 
#=============== Coarse-gain ===================

def get_word2vec_model(is_train, model_path, split_data=None, vector_size=None, min_count=None, window=None):
    """word2vec训练代码
    """
    if is_train:
        model = word2vec.Word2Vec(split_data, vector_size=vector_size, min_count=min_count, window=window)
        model.save(model_path)
    else:
        model = Word2Vec.load(model_path)
    return model


def get_coarse_simtiment_score(text, word2vec_model):
    word_seg = word_segment(text)
    sim_word = []
    sim_word_weight = []
    for e in word2vec_model.wv.most_similar(positive=word_seg, topn=10):
        # print(e[0], e[1])
        sim_word.append(e[0])
        sim_word_weight.append(e[1])

    return sim_word, softmax(sim_word_weight)

#============== End coarse-gain ================
#endregion

#region Merge-model Coarse-Fine 


#region Init 
#============== Init ===============
def CreateAndWriteCSV(name, data):
    if not os.path.exists("feature"):
        os.makedirs("feature")
    
    filename = os.path.join("feature", name + '.csv')
    # Mở file CSV để ghi
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Ghi tiêu đề
        writer.writerow(['Key', 'Array'])

        # Ghi dữ liệu từ từ điển vào file
        for key, value in data.items():
            # Kiểm tra kiểu dữ liệu trước khi chuyển đổi
            if isinstance(value, np.ndarray):
                array_str = str(value.tolist())
            else:
                array_str = str(value)
            writer.writerow([key, array_str])

    print(f'File CSV "{filename}" đã được tạo và ghi thành công.')
# Lấy dữ liệu của AB
data = read_data("./data/raw/All_Beauty_100reviews.json")
data_df = pd.DataFrame(data)
data_df.columns = ['reviewerID', 'asin', 'overall', 'reviewText']
data = data_df["reviewText"].tolist()
split_data = []
for i in data:
    split_data.append(word_segment(i))

# 1. (fine-gain) ------------------------

#  step1: LDA
num_topics = 10
num_words = 300
lda_model, dictionary, topic_to_words = get_lda_mdoel(split_data, num_topics, num_words)

# step2: 依存句法分析
model_path = './config/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'
parser_path = './config/stanford-parser-full-2020-11-17/stanford-parser.jar'

dep_parser = DependencyParser(model_path, parser_path)
# print(topic_to_words)
# step3: 情感词表
#----------------------------------------


# 2. (coarse_gain) ----------------------
# 粗粒度情感分析计算
# word2vec 参数设置
window_size = 3
min_count = 1
vector_size = 200
# model_path = "./output/word2vec.model"
model_path = "./output/word2vec.model"
is_train = True # 是否训练

# 新训练
word2vec_model = get_word2vec_model(is_train=is_train,
                           model_path=model_path,
                           split_data=split_data,
                           vector_size=vector_size,
                           min_count=min_count,
                           window=window_size)


def get_coarse_score(text, word2vec_model):
    """获取粗粒度评分
    """
    sim_word, sim_word_weight = get_coarse_simtiment_score(data[1], word2vec_model)
    score = 0
    for i, j in zip(sim_word, sim_word_weight):
        score += get_word_sentiment_score(i) * j
    return sigmoid(score)

#------------------------------------------

#================= End Init ===============
#endregion


# Cac feature co the am vi la danh gia tieu cuc
# 3. 粗细粒度融合
# Lấy đặc trưng chi tiết và đặc trưng thô của reviewer và item rồi tổng hợp ra được theta_u và theta_i
ErrorList = []
reviewer_feature_dict = {}
item_feature_dict = {}

reviewer_feature_dict_coarse = {}
item_feature_dict_coarse = {}
allFeatureReview = {}

def MergeFineCoarse_Reviewer(data_df, reviewer_feature_dict):
    print("=========================Merge Reviews==========================")
    count = 0
    for reviewer_id, df in data_df.groupby("reviewerID"):
        review_text = df["reviewText"].tolist()
        itemID = df["asin"].tolist()
        review_feature = 0
        coarse_feature = 0
        for i, text in enumerate(review_text):
            try:
                fine_feature = get_topic_sentiment_metrix(text, dictionary, lda_model, topic_to_words, dep_parser, topic_nums=num_topics) #Get score for each of related tocpic
                coarse_feature = get_coarse_score(text, word2vec_model)
                print("[",i ,"]", "Fine_feature: ", fine_feature, " - Coarse_feature: ", coarse_feature)
                allFeatureReview[i] = [reviewer_id, itemID, fine_feature, coarse_feature]
                print(allFeatureReview[i])
                if i == 0:
                    review_feature = fine_feature * coarse_feature
                else:
                    review_feature += fine_feature * coarse_feature
            except:
                print("Error: ", text)
                ErrorList.append(text)
                continue
        reviewer_feature_dict_coarse[reviewer_id] = coarse_feature
        reviewer_feature_dict[reviewer_id] = review_feature
        print(review_feature)

        
def MergeFineCoarse_Item(data_df, item_feature_dict):
    # 商品特征提取
    print("=========================Merge Items==========================")
    for asin, df in data_df.groupby("asin"):
        review_text = df["reviewText"].tolist()
        item_feature = 0 
        coarse_feature = 0
        for i, text in enumerate(review_text):
            try:
                fine_feature = get_topic_sentiment_metrix(text, dictionary, lda_model, topic_to_words, dep_parser, topic_nums=num_topics)
                coarse_feature = get_coarse_score(text, word2vec_model)

                print("[",i ,"]", "Fine_feature: ", fine_feature, " - Coarse_feature: ", coarse_feature)
                
                if i == 0:
                    item_feature = fine_feature * coarse_feature
                else:
                    item_feature += fine_feature * coarse_feature
            except:
                print("Error: ", text)
                ErrorList.append(text)
                continue
        item_feature_dict_coarse[asin] = coarse_feature
        item_feature_dict[asin] = item_feature
        print(item_feature)

def load_data_from_csv(file_path):
    data = {}
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            key = row['Key']
            array_str = row['Array']
            # Chuyển chuỗi thành mảng Python
            array = np.array(ast.literal_eval(array_str))
            data[key] = array
    return data
# if os.path.exists("feature/review_feature.csv") and os.path.exists("feature/item_feature.csv"):
#     reviewer_feature_dict = load_data_from_csv("feature/review_feature.csv")
#     item_feature_dict = load_data_from_csv("feature/item_feature.csv")
# else:
#     MergeFineCoarse_Item_Reviewer(data_df, reviewer_feature_dict, item_feature_dict)
#     print(reviewer_feature_dict)
#     CreateAndWriteCSV('review_feature', reviewer_feature_dict)
#     print(item_feature_dict)
#     CreateAndWriteCSV('item_feature', item_feature_dict)

if os.path.exists("feature_backup/review_feature.csv"):
    reviewer_feature_dict = load_data_from_csv("feature/review_feature.csv")
    # reviewer_feature_dict_coarse = load_data_from_csv("feature_backup/review_feature_coarse.csv")
else:
    MergeFineCoarse_Reviewer(data_df, reviewer_feature_dict)
    print(reviewer_feature_dict)
    CreateAndWriteCSV('review_feature', reviewer_feature_dict)
    CreateAndWriteCSV('review_feature_coarse', reviewer_feature_dict_coarse)

if os.path.exists("feature_backup/item_feature.csv"):
    item_feature_dict = load_data_from_csv("feature/item_feature.csv")
    # item_feature_dict_coarse = load_data_from_csv("feature_backup/item_feature_coarse.csv")
else:
    MergeFineCoarse_Item(data_df, item_feature_dict)
    print(item_feature_dict)
    CreateAndWriteCSV('item_feature', item_feature_dict)
    CreateAndWriteCSV('item_feature_coarse', item_feature_dict_coarse)
    
print(allFeatureReview)
 #===================================
 

print(reviewer_feature_dict['A1118RD3AJD5KH'])
#endregion

#region visualize_feature_vectors and reduce outliers
# import pandas as pd
# import ast
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.spatial import distance

# def read_csv_to_dict(file_path):
#     data_dict = {}
#     with open(file_path, 'r', newline='') as file:
#         csv_reader = csv.reader(file)
#         for row in csv_reader:
#             try:
#                 if len(row) >= 2:
#                     key = row[0]
#                     value = ast.literal_eval(row[1])  # Chuyển đổi chuỗi thành list
#                     data_dict[key] = value
#             except Exception as e:
#                 print(f"Error reading line: {row}. Error: {e}")
#     return data_dict


# def visualize_feature_vectors(file_path):
#     # Đọc file CSV
#     feature_dict = read_csv_to_dict(file_path)
    

#     # Chuyển đổi chuỗi trong cột 'Array' thành list
#     # df['Array'] = df['Array'].apply(ast.literal_eval)

#     # Lấy vector đặc trưng từ cột 'Array'
#     # feature_vectors = df['Array'].tolist()
#     # Vector 0
#     zero_vector = [1.0] * 10
#     feature_distance = []
#     for key, value in feature_dict.items():
#         cosine_distance = distance.cosine(value, zero_vector)
#         feature_distance.append(cosine_distance)
#     # Biểu diễn lên boxplot sử dụng seaborn
#     print(len(feature_distance))
#     # plt.figure(figsize=(10, 6))
#     # sns.boxplot(data=feature_distance)
#     # plt.title('Boxplot of Feature Vectors')
#     # plt.xlabel('Feature Dimension')
#     # plt.ylabel('Feature Value')

#     # Loại bỏ outliers
#     Q1 = np.percentile(feature_distance, 25, axis=0)
#     Q3 = np.percentile(feature_distance, 75, axis=0)
#     IQR = Q3 - Q1

#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     # Lọc và loại bỏ outliers
#     filtered_feature_vectors = np.array(feature_distance)
#     filtered_feature_vectors[(feature_distance < lower_bound) | (feature_distance > upper_bound)] = np.nan
    
#     filtered_feature_dict = {key: value for key, value in feature_dict.items() if lower_bound <= distance.cosine(value, zero_vector) <= upper_bound}
#     print(len(filtered_feature_dict))
#     # Biểu diễn boxplot của tập hợp đã lọc
#     # plt.figure(figsize=(10, 6))
#     # sns.boxplot(data=filtered_feature_vectors)
#     # plt.title('Boxplot of Filtered Feature Vectors')
#     # plt.xlabel('Feature Dimension')
#     # plt.ylabel('Feature Value')

#     # plt.show()
#     return filtered_feature_dict

# # Thực hiện hàm với đường dẫn đến file CSV của bạn
# reviewer_feature_dict = visualize_feature_vectors('feature_backup/review_feature.csv')  # Thay 'your_file.csv' bằng đường dẫn thực tế của file CSV của bạn
# item_feature_dict = visualize_feature_vectors('feature_backup/item_feature.csv')  # Thay 'your_file.csv' bằng đường dẫn thực tế của file CSV của bạn

# CreateAndWriteCSV('review_feature', reviewer_feature_dict)
# CreateAndWriteCSV('item_feature', item_feature_dict)
#endregion

#=======================================================================

def direct_sum(A, B):
    m, n = A.shape
    p, q = B.shape
    result = np.zeros((m+p, n+q), dtype=A.dtype)
    result[:m, :n] = A
    result[m:, n:] = B
    return result

#region SVDFeature
import svd
from svd import SVD
import torch

checkpoint_path = 'chkpt/svd.pt'
# Kiểm tra xem tệp checkpoint có tồn tại không
if os.path.exists(checkpoint_path):
    # Tải mô hình từ checkpoint
    svd = torch.load(checkpoint_path)
    
else:
    svd = SVD()
    svd.train() 
    torch.save(svd, checkpoint_path)
emb_user,emb_item = svd.get_embedings()
# print(emb_user)
# print(emb_item)
print(emb_user.shape)
print(emb_item.shape)
# print(np.sqrt(svd.cost(emb_user,emb_item)))
print(len(reviewer_feature_dict))
print(len(item_feature_dict))
# print(svd.get_user_embedding('AX0ZEGHH0H525').shape)


#=================================== Merge to z ======================================
def read_csv_file(csv_file):
    """
    Đọc tập tin CSV với ID là một chuỗi và giá trị là một vector đặc trưng.

    Args:
    - csv_file: Đường dẫn đến tập tin CSV

    Returns:
    - keys: Danh sách các key (chuỗi)
    - values: Danh sách các value (vector)
    """
    keys = []
    values = []

    with open(csv_file, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Bỏ qua header
        for row in csv_reader:
            key = row[0]  # ID là cột đầu tiên
            value_str = row[1]  # Giá trị là cột thứ hai
            value = ast.literal_eval(value_str)  # Chuyển đổi chuỗi thành vector
            keys.append(key)
            values.append(np.array(value))

    return keys, values
    
def mergeReview_Rating(path, filename, getEmbedding):
    reviewerID,_ = read_csv_file(path)
    feature_dict = {}
    z_list = []
    for feature in reviewerID:
        if getEmbedding == "reviewer":
            A = reviewer_feature_dict[feature]
            B = svd.get_user_embedding(feature)
        else:
            A = item_feature_dict[feature]
            B = svd.get_item_embedding(feature)
        z = np.concatenate((np.array(A), np.array(B)))
        # z = np.array(A) + np.array(B)
        feature_dict[feature] = z
        # print(np.sum(z))
        z_list.append(z)
    CreateAndWriteCSV(filename, feature_dict)
    # return z_list
    return feature_dict
    
    

# z_item = mergeReview_Rating("feature_backup/item_feature.csv", "z_item", "item")
# z_review = mergeReview_Rating("feature_backup/review_feature.csv", "z_reviewer", "reviewer")
z_item = mergeReview_Rating("feature_backup/item_feature.csv", "z_item", "item")
z_review = mergeReview_Rating("feature_backup/review_feature.csv", "z_reviewer", "reviewer")
# print(z_item)
# print(z_review)

# print(z_review)

#==============================================================================

#============================ Calulate U/I deep ===============================

def Caculate_Deep(v, z):
    """
    Tính Udeep theo công thức trong hình.

    Tham số:
        v: List các giá trị v.
        z: List các giá trị z.

    Trả về:
        Giá trị Udeep.
    """
    list_sum = {}
    i = 0
    for name, z_i in z.items():
        if i < len(v):  # Đảm bảo vẫn còn phần tử trong danh sách v
            v_i = v[i]
            sum_v_z = v_i * z_i
            sum_v2_z2 = (v_i**2) * (z_i**2)
            result = (1 / 2) * ((sum_v_z)**2 - sum_v2_z2)
            list_sum[name] = result
    return list_sum

#==============================================================================

from train import *
from config import args
from data_process import *


device = torch.device(args.device)
dataset = get_dataset(args.dataset_name, args.dataset_path)
model = get_model(dataset).to(device)
v_list = np.array(model.embedding.embedding.weight.data.tolist())

u_deep = Caculate_Deep(v_list, z_review)
i_deep = Caculate_Deep(v_list, z_item)
CreateAndWriteCSV("u_deep", u_deep)
CreateAndWriteCSV("i_deep", i_deep)

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


# Sử dụng function merge_csv_columns
merge_csv_columns('data/ratings_AB.csv', 'reviewerID', 'transformed_udeep.csv', 'ID', 'Array', 'Udeep')
merge_csv_columns('data/ratings_AB.csv', 'asin', 'transformed_ideep.csv', 'ID', 'Array', 'Ideep')

#endregion


# import numpy as np

# class FactorizationMachine(object):

#     def __init__(self, n_features, n_factors):
#         self.n_features = n_features
#         self.n_factors = n_factors

#         # W_0 là trọng số của term đại diện cho bias
#         self.W_0 = np.zeros(1)

#         # W_i là trọng số của các term đại diện cho các feature riêng lẻ
#         self.W_i = np.zeros((n_features, n_factors))

#         # V_ij là trọng số của các term đại diện cho các interaction giữa các feature
#         self.V_ij = np.zeros((n_features, n_features, n_factors))

#     def fit(self, X, y):
#         """
#         Fits the model to the given data.

#         Args:
#             X: The input data, a 2D NumPy array of shape (n_samples, n_features).
#             y: The target values, a 1D NumPy array of shape (n_samples,).

#         Returns:
#             None
#         """

#         # Khởi tạo các biến cần thiết
#         self.W_0 = np.zeros(1)
#         self.W_i = np.zeros((self.n_features, self.n_factors))
#         self.V_ij = np.zeros((self.n_features, self.n_features, self.n_factors))

#         # Tính toán các gradient
#         gradients = self.gradient(X, y)

#         # Cập nhật các trọng số
#         self.W_0 += gradients[0]
#         self.W_i += gradients[1]
#         self.V_ij += gradients[2]

#     def predict(self, X):
#         """
#         Predicts the target values for the given data.

#         Args:
#             X: The input data, a 2D NumPy array of shape (n_samples, n_features).

#         Returns:
#             The predicted target values, a 1D NumPy array of shape (n_samples,).
#         """

#         # Tính toán output của model
#         output = self.W_0
#         for i in range(self.n_features):
#             output += np.sum(self.W_i[i] * np.power(X[:, i], 2))
#             for j in range(i + 1, self.n_features):
#                 output += np.sum(self.V_ij[i, j] * X[:, i] * X[:, j])

#         return output

#     def gradient(self, X, y):
#         """
#         Calculates the gradients of the loss function with respect to the model parameters.

#         Args:
#             X: The input data, a 2D NumPy array of shape (n_samples, n_features).
#             y: The target values, a 1D NumPy array of shape (n_samples,).

#         Returns:
#             The gradients, a 3D NumPy array of shape (3, n_features, n_factors).
#         """

#         # Tính toán output của model
#         output = self.predict(X)

#         # Tính toán các gradient
#         gradients = np.zeros((3, self.n_features, self.n_factors))
#         gradients[0] = (output - y).reshape((X.shape[0], 1))
#         for i in range(self.n_features):
#             gradients[1][i] = (output - y) * 2 * X[:, i].reshape((X.shape[0], 1))
#             for j in range(i + 1, self.n_features):
#                 gradients[2][i, j] = (output - y) * X[:, i] * X[:, j].reshape((X.shape[0], 1))

#         return gradients
#     def train_fm(model, X, y, n_epochs, learning_rate):
#         for epoch in range(n_epochs):
#             for i in range(X.shape[0]):
#                 gradients = model.gradient(X[i], y[i])
#                 model.W_0 += learning_rate * gradients[0]
#                 model.W_i += learning_rate * gradients[1]
#                 model.V_ij += learning_rate * gradients[2]

#         return model
#     def test(model, data_loader, device):
#         model.eval()
#         targets, predicts = list(), list()
#         with torch.no_grad():
#             for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
#                 fields, target = fields.to(device), target.to(device)
#                 y = model(fields)
#                 targets.extend(target.tolist())
#                 predicts.extend(y.tolist())
        
#         return roc_auc_score(targets, predicts)
    
# import torch
# import tqdm
# from config import args
# from sklearn.metrics import roc_auc_score
# from data_process import ReviewAmazon
# from torch.utils.data import DataLoader
# from torchfm.model.fm import FactorizationMachineModel

# def get_dataset(name, path):
#     if name == 'reviewAmazon':
#         return ReviewAmazon(path)
    
# def get_model(dataset):
#     field_dims = dataset.field_dims
#     print("dataset_shape: ", len(dataset))
#     return FactorizationMachineModel(field_dims, embed_dim=16)

# dataset = get_dataset(args.dataset_name, args.dataset_path)
# train_length = int(len(dataset) * 0.7)
# valid_length = int(len(dataset) * 0.1)
# test_length = len(dataset) - train_length - valid_length
# train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
#     dataset, (train_length, valid_length, test_length))
# train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8)
# valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=8)
# test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8)

# # # Train model
# # model = get_model(dataset)
# # model = model.train_fm(model, X_train, y_train, n_epochs=100, learning_rate=0.01)

# # # Đánh giá model
# # auc = model.test()
# #endregion


# region group by id
# import json

# def group_by_ids(dataset):
#     reviewer_to_items = {}
#     item_to_reviewers = {}
    
#     for review in dataset:
#         reviewer_id = review["reviewerID"]
#         asin_id = review["asin"]
        
#         # Thêm reviewer_id vào danh sách các items của reviewer
#         if reviewer_id not in reviewer_to_items:
#             reviewer_to_items[reviewer_id] = []
#         reviewer_to_items[reviewer_id].append(asin_id)
        
#         # Thêm asin_id vào danh sách các reviewers của item
#         if asin_id not in item_to_reviewers:
#             item_to_reviewers[asin_id] = []
#         item_to_reviewers[asin_id].append(reviewer_id)
    
#     return reviewer_to_items, item_to_reviewers

# # Đọc dữ liệu từ tập dataset
# with open("data/All_Beauty_5.json") as f:
#     dataset = [json.loads(line) for line in f]

# # Gom nhóm theo reviewerID và asinID
# reviewer_to_items, item_to_reviewers = group_by_ids(dataset)

# # In kết quả
# print("Danh sách các items của mỗi reviewer:")
# print(len(reviewer_to_items))
# # for reviewer_id, items in reviewer_to_items.items():
# #     print(f"Reviewer ID: {reviewer_id}, Items: {items}")

# print(len(item_to_reviewers))
# print("\nDanh sách các reviewers của mỗi item:")
# # for asin_id, reviewers in item_to_reviewers.items():
#     # print(f"ASIN ID: {asin_id}, Reviewers: {reviewers}")
    
#endregion

#region get bias
# import json

# def average_rating_by_reviewer(reviewer_id, json_file):
#     """
#     Tính trung bình rating của các mục mà một người dùng đã đánh giá.

#     Args:
#     - reviewer_id: ID của người dùng (reviewer).
#     - json_file: Đường dẫn đến tệp JSON chứa tập dữ liệu.

#     Returns:
#     - Trung bình rating của các mục mà người dùng đã đánh giá.
#     """
#     total_rating = 0
#     count = 0

#     with open(json_file) as f:
#         dataset = [json.loads(line) for line in f]

#     for review in dataset:
#         if review['reviewerID'] == reviewer_id:
#             total_rating += review['overall']
#             count += 1

#     if count == 0:
#         return None  # Trường hợp không có đánh giá từ người dùng này

#     average_rating = total_rating / count
#     return average_rating 

# def average_rating_by_item(item_id, json_file):
#     """
#     Tính trung bình rating của một mục (item) dựa trên itemID.

#     Args:
#     - item_id: ID của mục (item) cần tính trung bình rating.
#     - json_file: Đường dẫn đến tệp JSON chứa tập dữ liệu.

#     Returns:
#     - Trung bình rating của mục (item) cần tính.
#     """
#     total_rating = 0
#     count = 0

#     with open(json_file) as f:
#         dataset = [json.loads(line) for line in f]
        
#     for review in dataset:
#         if review['asin'] == item_id:
#             total_rating += review['overall']
#             count += 1

#     if count == 0:
#         return None  # Trường hợp không có đánh giá cho mục này

#     average_rating = total_rating / count
#     return average_rating

# # Sử dụng hàm để tính trung bình rating của một mục cụ thể
# json_file_path = "data/All_Beauty_5.json"
# item_id = "B00006L9LC"
# average_rating = average_rating_by_item(item_id, json_file_path)
# if average_rating is not None:
#     print(f"Trung bình rating của mục {item_id} là: {average_rating}")
# else:
#     print(f"Mục {item_id} chưa được đánh giá.")

# reviewer_id = "AEL1DK2OJ41ZZ"
# average_rating = average_rating_by_reviewer(reviewer_id, json_file_path)
# if average_rating is not None:
#     print(f"Trung bình rating của người dùng {reviewer_id} là: {average_rating}")
# else:
#     print(f"Người dùng {reviewer_id} chưa đánh giá bất kỳ mục nào.")

#endregion