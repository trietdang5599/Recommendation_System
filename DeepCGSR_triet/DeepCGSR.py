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
from config import args
# nltk.download('wordnet')
# nltk.download('sentiwordnet')

isRemoveOutliner = args.isRemoveOutliner
dataset_json = args.dataset_json
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
# Lấy dữ liệu của AB
data = read_data(dataset_json)
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
    sim_word, sim_word_weight = get_coarse_simtiment_score(text, word2vec_model)
    score = 0
    for i, j in zip(sim_word, sim_word_weight):
        score += get_word_sentiment_score(i) * j
    return sigmoid(score)

#------------------------------------------

#================= End Init ===============
#endregion
import re

def remove_special_characters(text):
    # Sử dụng biểu thức chính quy để loại bỏ các ký tự đặc biệt
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

# Cac feature co the am vi la danh gia tieu cuc
# 3. 粗细粒度融合
# Lấy đặc trưng chi tiết và đặc trưng thô của reviewer và item rồi tổng hợp ra được theta_u và theta_i
ErrorList = []
reviewer_feature_dict = {}
item_feature_dict = {}

reviewer_feature_dict_coarse = {}
item_feature_dict_coarse = {}
allFeatureReview = pd.DataFrame(columns=['reviewerID', 'itemID', 'overall', 'unixReviewTime', 'fine_feature', 'coarse_feature'])
rowList = []

def ExtractReviewFeature(data_df):
    # 商品特征提取
    print("=========================Merge Items==========================")
    for asin, df in tqdm.tqdm(data_df.groupby("asin")):
        review_text = df["reviewText"].tolist()
        reviewerID = df["reviewerID"].tolist()
        overall = df["overall"].tolist()
        coarse_feature = 0
        for i, text in enumerate(review_text):
            try:
                # text = text[:500]
                # text = remove_special_characters(text)
                fine_feature = get_topic_sentiment_metrix(text, dictionary, lda_model, topic_to_words, dep_parser, topic_nums=num_topics)
                coarse_feature = get_coarse_score(text, word2vec_model)
                print("[",i ,"]", "Fine_feature: ", fine_feature, " - Coarse_feature: ", coarse_feature)
                new_row = {'reviewerID':reviewerID[i], 'itemID':asin, 'overall': overall[i],
                           'fine_feature':fine_feature, 'coarse_feature':coarse_feature}
                rowList.append(new_row)
            except Exception as e:
                print("Error: ", e, "Text: ", text)
                ErrorList.append(text)
                continue
    return pd.DataFrame(rowList, columns=['reviewerID', 'itemID', 'overall', 'fine_feature', 'coarse_feature'])
        
def MergeFineCoarseFeatures(data_df, groupBy="reviewerID"):
    print("=========================Merge Features==========================")
    feature_dict = {}
    for id, df in data_df.groupby(groupBy):
        count = 0
        feature = np.zeros(10) # Khởi tạo vector đặc trưng
        list_finefeature = df['fine_feature']
        list_coarse_feature = df['coarse_feature']
        for fine, coarse in zip(list_finefeature, list_coarse_feature):
            try:
                fine_feature = np.fromstring(fine.strip('[]'), dtype=float, sep=' ')
                coarse_feature = float(coarse)
                if count == 0:
                    feature = fine_feature * coarse_feature
                else:
                    feature += fine_feature * coarse_feature
                count += 1
            except Exception as e:
                print("Error: ", e)
                continue
        feature_dict[id] = np.array(feature.tolist())  
    return feature_dict



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


if os.path.exists("./feature/allFeatureReview.csv"):
    allFeatureReview = pd.read_csv("./feature/allFeatureReview.csv")
    # reviewer_feature_dict_coarse = load_data_from_csv("feature_backup/review_feature_coarse.csv")
else:
    allFeatureReview = ExtractReviewFeature(data_df)
    allFeatureReview.to_csv('./feature/allFeatureReview.csv', index=False)
    
# print(allFeatureReview)





# Hàm chuyển đổi chuỗi thành list các số thực
def string_to_list(string):
    string = string.strip('[]')  # Loại bỏ ký tự '[' và ']' ở đầu và cuối chuỗi
    if string:
        return [float(x.strip()) for x in string.split(',')]
    else:
        return []


 #region remove outliner
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

averageVectorbyItem = {}



# def reduce_dimensionality(matrix, method='pca', n_components=1):
#     if method == 'pca':
#         pca = PCA(n_components=n_components)
#         reduced_matrix = pca.fit_transform(matrix)
#     elif method == 'tsne':
#         tsne = TSNE(n_components=n_components)
#         reduced_matrix = tsne.fit_transform(matrix)
#     elif method == 'umap':
#         reducer = umap.UMAP(n_components=n_components)
#         reduced_matrix = reducer.fit_transform(matrix)
#     else:
#         raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'.")
#     return reduced_matrix.reshape(1, -1)

def list_to_matrix(vector_list):
    matrix = np.vstack(vector_list)
    return matrix

def AverageVector():
    data_df = pd.read_csv("./feature/allFeatureReview.csv")
    
    data_df.columns = ['ReviewerID', 'itemID', 'overall', 'fine_feature', 'coarse_feature']
    averageVector = np.zeros(10)
    for key, value in data_df.groupby('itemID'):
        # Chuyển đổi kiểu dữ liệu của 'fine_feature' từ chuỗi sang mảng numpy số thực
        value['fine_feature'] = value['fine_feature'].apply(lambda x: np.fromstring(x.strip('[]'), dtype=float, sep=' '))
        
        # Chuyển đổi kiểu dữ liệu của 'coarse_feature' từ chuỗi sang số thực
        value['coarse_feature'] = value['coarse_feature'].apply(lambda x: float(x))
        
        # Thực hiện phép nhân giữa 'fine_feature' và 'coarse_feature'
        list_feature = [np.multiply(a, b) for a, b in zip(value['fine_feature'], value['coarse_feature'])]
        
        # Tính vector trung bình
        # averageVector = np.mean(list_feature, axis=0)
        
        maxtrix_feature = list_to_matrix(list_feature)
        # print("maxtrix: ", maxtrix_feature)
        # averageVector = reduce_dimensionality(maxtrix_feature, method='umap', n_components=1)
        averageVectorbyItem[key] = averageVector.tolist()
    # print("average: ", averageVectorbyItem)
    CreateAndWriteCSV('averageVectorbyItem', averageVectorbyItem)
AverageVector()

filteredFeatures = pd.DataFrame(columns=['reviewerID', 'itemID', 'overall', 'fine_feature', 'coarse_feature'])

def remove_outliers(filepath, listAverageVector):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(filepath)
    result =pd.DataFrame(columns=['reviewerID', 'itemID', 'overall', 'fine_feature', 'coarse_feature'])
    for itemID, df in data.groupby('itemID'):
        list_finefeature = df['fine_feature']
        list_coarse_feature = df['coarse_feature']
        feature_distance = []
        normalize_vector = listAverageVector[itemID]
        for fine, coarse in zip(list_finefeature, list_coarse_feature):
            fine_feature = np.fromstring(fine.strip('[]'), dtype=float, sep=' ')
            coarse_feature = float(coarse)
            feature = fine_feature * coarse_feature
            
            cosine_distance = distance.cosine(feature, normalize_vector)
            feature_distance.append(cosine_distance)
        Q1 = np.percentile(feature_distance, 25, axis=0)
        Q3 = np.percentile(feature_distance, 75, axis=0)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Tạo mảng boolean cho việc loại bỏ ngoại lai
        outliers_mask = np.array([lower_bound <= x <= upper_bound for x in feature_distance])
        # Loại bỏ các giá trị ngoại lai
        df.drop(df[~outliers_mask].index, inplace=True)
        # Thêm vào result
        result = pd.concat([result, df], axis=0, ignore_index=True)
    return result
filteredFeatures = remove_outliers('./feature/allFeatureReview.csv', averageVectorbyItem)
filteredFeatures.to_csv('feature/filteredFeatures.csv', index=False)

# print(reviewer_feature_dict)
# print(item_feature_dict)
#endregion

#region mergeReviewFeatures
if os.path.exists("./feature/reviewer_feature.csv"):
    reviewer_feature_dict = load_data_from_csv("./feature/reviewer_feature.csv")
else:
    if isRemoveOutliner:
        # use filtered data
        reviewer_feature_dict = MergeFineCoarseFeatures(filteredFeatures, groupBy="reviewerID")
    else:
        # use filtered data
        reviewer_feature_dict = MergeFineCoarseFeatures(allFeatureReview, groupBy="reviewerID")
    CreateAndWriteCSV('reviewer_feature', reviewer_feature_dict)
    
if os.path.exists("./feature/item_feature.csv"):
    item_feature_dict = load_data_from_csv("./feature/item_feature.csv")
else:
    if isRemoveOutliner:
        # use filtered data
        item_feature_dict = MergeFineCoarseFeatures(filteredFeatures, groupBy="itemID")
    else:
        # use all data
        item_feature_dict = MergeFineCoarseFeatures(allFeatureReview, groupBy="itemID")
    CreateAndWriteCSV('item_feature', item_feature_dict)
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
    if isRemoveOutliner:
        svd = SVD('feature/filteredFeatures.csv')
    else:
        svd = SVD('feature/allFeatureReview.csv')
    # svd.data_path = 'feature/allFeatureReview.csv'
    svd.train() 
    torch.save(svd, checkpoint_path)
emb_user,emb_item = svd.get_embedings()
# print(emb_user)
# print(emb_item)
# print(emb_user.shape)
# print(emb_item.shape)
# print(np.sqrt(svd.cost(emb_user,emb_item)))
print(len(reviewer_feature_dict))
args.user_length = len(reviewer_feature_dict)
print(len(item_feature_dict))
args.item_length = len(item_feature_dict)
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
    for id in reviewerID:
        if getEmbedding == "reviewer":
            A = reviewer_feature_dict[id]
            B = svd.get_user_embedding(id)
        else:
            A = item_feature_dict[id]
            B = svd.get_item_embedding(id)

        z = np.concatenate((np.array(A), np.array(B)))
        # z = np.array(A) + np.array(B)
        feature_dict[id] = z
        # print(np.sum(z))
        z_list.append(z)
    CreateAndWriteCSV(filename, feature_dict)
    # return z_list
    return feature_dict
 
z_item = mergeReview_Rating("feature/item_feature.csv", "z_item", "item")
z_review = mergeReview_Rating("feature/reviewer_feature.csv", "z_reviewer", "reviewer")
# print("z_item: ",z_item)
# print("z_review: ", z_review)

#==============================================================================

#============================ Calulate U/I deep ===============================


def Calculate_Deep(v, z, start):
    list_sum = {}
    i = start
    for name, z_i in z.items():
        if i < len(v):  # Đảm bảo vẫn còn phần tử trong danh sách v
            v_i = v[i]
            sum_v_z = v_i * z_i
            sum_v2_z2 = (v_i**2) * (z_i**2)
            result = (1 / 2) * ((sum_v_z)**2 - sum_v2_z2)
            list_sum[name] = result
            i += 1
    return list_sum

#==============================================================================

from train import *
from config import args
from data_process import *


# device = torch.device(args.device)
dataset = get_dataset(args.dataset_name, args.data_feature)
# model = get_model(dataset).to(device)
model = get_model(dataset)
v_list = np.array(model.embedding.embedding.weight.data.tolist())
# print(v_list)
u_deep = Calculate_Deep(v_list, z_review, 0)
i_deep = Calculate_Deep(v_list, z_item, len(z_review))
CreateAndWriteCSV("u_deep", u_deep)
CreateAndWriteCSV("i_deep", i_deep)
TransformLabel_Deep(pd.read_csv("feature/u_deep.csv", sep=',', engine='c', header='infer').to_numpy()[:, :3], "feature/transformed_udeep.csv")
TransformLabel_Deep(pd.read_csv("feature/i_deep.csv", sep=',', engine='c', header='infer').to_numpy()[:, :3], "feature/transformed_ideep.csv")


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
merge_csv_columns(args.data_feature, 'reviewerID', 'feature/u_deep.csv', 'Key', 'Array', 'Udeep')
merge_csv_columns(args.data_feature, 'itemID', 'feature/i_deep.csv', 'Key', 'Array', 'Ideep')

#endregion

