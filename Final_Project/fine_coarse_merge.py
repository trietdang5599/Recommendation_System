import os
import numpy as np
import pandas as pd
from fine_gain import get_lda_mdoel, DependencyParser, get_word_sentiment_score, get_topic_sentiment_metrix
from coarse_gain import get_word2vec_model, get_coarse_simtiment_score
import tqdm
from utils import read_data, word_segment, sigmoid, load_data_from_csv, CreateAndWriteCSV
import nltk
from config import args
# nltk.download('wordnet')
# nltk.download('sentiwordnet')

isRemoveOutliner = args.isRemoveOutliner
dataset_json = args.dataset_json

#region Init 
#============== Init ===============

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
ErrorList = []
reviewer_feature_dict = {}
item_feature_dict = {}
reviewer_feature_dict_coarse = {}
item_feature_dict_coarse = {}
allFeatureReview = pd.DataFrame(columns=['reviewerID', 'itemID', 'overall', 'unixReviewTime', 'fine_feature', 'coarse_feature'])
rowList = []
#================= End Init ===============

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
def GetReviewFeatures(file_path = "./feature/allFeatureReview.csv"):
    if os.path.exists(file_path):
        allFeatureReview = pd.read_csv(file_path)
        # reviewer_feature_dict_coarse = load_data_from_csv("feature_backup/review_feature_coarse.csv")
    else:
        allFeatureReview = ExtractReviewFeature(data_df)
        allFeatureReview.to_csv(file_path, index=False)        
    return allFeatureReview
# allFeatureReview = GetReviewFeatures()

def MergeFineCoarseFeatures(data_df, groupBy="reviewerID"):
    print("=========================Merge Features==========================")
    feature_dict = {}
    for id, df in data_df.groupby(groupBy):
        
        feature = np.zeros(10) # Khởi tạo vector đặc trưng
        list_finefeature = df['fine_feature']
        list_coarse_feature = df['coarse_feature']
        print(list_coarse_feature)
        for fine, coarse in zip(list_finefeature, list_coarse_feature):
            count = 0
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
        print(feature)
        feature_dict[id] = np.array(feature.tolist())  
    return feature_dict

def GetReview_ItemFeatures(allFeatureReview, file_ReviewerFeature_path = "./feature/reviewer_feature.csv", file_ItemFeature_path = "./feature/item_feature.csv"):
    if os.path.exists(file_ReviewerFeature_path):
        reviewer_feature_dict = load_data_from_csv(file_ReviewerFeature_path)
    else:
        # use filtered data
        reviewer_feature_dict = MergeFineCoarseFeatures(allFeatureReview, groupBy="reviewerID")
        CreateAndWriteCSV('reviewer_feature', reviewer_feature_dict)
        
    if os.path.exists(file_ItemFeature_path):
        item_feature_dict = load_data_from_csv(file_ItemFeature_path)
    else:
        # use all data
        item_feature_dict = MergeFineCoarseFeatures(allFeatureReview, groupBy="itemID")
        CreateAndWriteCSV('item_feature', item_feature_dict)
    return reviewer_feature_dict, item_feature_dict

# reviewer_feature_dict, item_feature_dict = GetReview_ItemFeatures()
