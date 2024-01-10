# -*- coding:utf-8 -*-
import csv
import pandas as pd
from utils import read_data, sigmoid, word_segment
from coarse_gain import get_word2vec_model, get_coarse_simtiment_score
from fine_gain import get_lda_mdoel, DependencyParser, get_word_sentiment_score, get_topic_sentiment_metrix

def CreateAndWriteCSV(name, data):
    filename = name + '.csv'
    # Mở file CSV để ghi
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Ghi tiêu đề
        writer.writerow(['Key', 'Array'])

        # Ghi dữ liệu từ từ điển vào file
        for key, value in data.items():
            writer.writerow([key, value])

    print(f'File CSV "{filename}" đã được tạo và ghi thành công.')

# 数据读取
data = read_data("./data/All_Beauty_5.json")
data_df = pd.DataFrame(data)
data_df.columns = ['reviewerID', 'asin', 'overall', 'reviewText']
data = data_df["reviewText"].tolist()
split_data = []
for i in data:
    split_data.append(word_segment(i))

# 1. 细粒度情感分析 (fine-gain)

#  step1: LDA
num_topics = 10
num_words = 300
lda_model, dictionary, topic_to_words = get_lda_mdoel(split_data, num_topics, num_words)

# step2: 依存句法分析
model_path = './config/stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'
parser_path = './config/stanford-parser-full-2020-11-17/stanford-parser.jar'

dep_parser = DependencyParser(model_path, parser_path)

print("====================TOPICS=======================")
print(topic_to_words)
print("=================================================")
# print(dictionary)
# step3: 情感词表


# 2.粗粒度情感分析 (coarse_gain)
# 粗粒度情感分析计算
# word2vec 参数设置
window_size = 3
min_count = 1
vector_size = 200
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

# Cac feature co the am vi la danh gia tieu cuc
# 3. 粗细粒度融合
# 用户特征提取
ErrorList = []

reviewer_feature_dict = {}
for reviewer_id, df in data_df.groupby("reviewerID"):
    review_text = df["reviewText"].tolist()
    for i, text in enumerate(review_text):
        try:
            #print("Text: ", text)
            fine_feature = get_topic_sentiment_metrix(text, dictionary, lda_model, topic_to_words, dep_parser, topic_nums=num_topics) #Get score for each of related tocpic
            coarse_feature = get_coarse_score(text, word2vec_model)
            print("[",i ,"]", "Fine_feature: ", fine_feature, " - Coarse_feature: ", coarse_feature)
            if i == 0:
                review_feature = fine_feature * coarse_feature
            else:
                review_feature += fine_feature * coarse_feature
        except:
            print("Error: ", text)
            ErrorList.append(text)
            continue
    reviewer_feature_dict[reviewer_id] = review_feature
    print(review_feature)

print("===================================")
# 商品特征提取
item_feature_dict = {}
for asin, df in data_df.groupby("asin"):
    review_text = df["reviewText"].tolist()
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
        
        
    item_feature_dict[asin] = item_feature
    print(item_feature)

print(reviewer_feature_dict)
CreateAndWriteCSV('review_feature', reviewer_feature_dict)
print(item_feature_dict)
CreateAndWriteCSV('item_feature', item_feature_dict)
print("保存用户矩阵个数：{}".format(len(reviewer_feature_dict))) 
print("保存商品矩阵个数：{}".format(len(item_feature_dict)))

