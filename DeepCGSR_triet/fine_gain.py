# -*- coding:utf-8 -*-
import torch
from lda_bert import LDA_BERT
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import sentiwordnet as swn
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from utils import word_segment, preprocessed



def get_lda_mdoel(split_data, num_topics, num_words):
    """ LDA模型训练词表构建主题单词矩阵获取
    """

    # 构建词表
    dictionary = corpora.Dictionary(split_data)
    corpus = [dictionary.doc2bow(text) for text in split_data]

    # LDA模型训练
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    # path = "./data/All_Beauty_5.json"
    # data = pd.read_json(path, lines=True)
    # count = 0
    # index = []
    # for i in range(len(data)):
    # #print(i)
    #     if type(data.iloc[i, 7]) == float:
    #         count += 1
    #     else:
    #         index.append(i)

    # documents = data.iloc[index, 7]
    # documents = documents.reset_index()
    # documents.drop('index', inplace = True, axis = 1)

    # # create data frame with all abstracts, use as input corpus
    # documents['index'] = documents.index.values
    # model = LDA_BERT(k = num_topics, method = "LDA_BERT", documents=documents)
    # 主题单词矩阵
    topic_to_words = []
    for i in range(num_topics):
        cur_topic_words = [ele[0] for ele in model.show_topic(i, num_words)]
        topic_to_words.append(cur_topic_words)
    return model, dictionary, topic_to_words

def get_tbert_model(split_data, num_topics, num_words):
    """ T-BERT模型训练词表构建主题单词矩阵获取 """
    
    # Load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Tokenize and get BERT embeddings for each document
    def get_bert_embeddings(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    embeddings = [get_bert_embeddings([' '.join(text)]) for text in split_data]
    embeddings = torch.vstack(embeddings).numpy()
    
    # Clustering to find topics
    kmeans = KMeans(n_clusters=num_topics, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    
    # Extract top words for each topic
    topic_to_words = []
    for i in range(num_topics):
        cluster_indices = [j for j, label in enumerate(labels) if label == i]
        cluster_texts = [' '.join(split_data[j]) for j in cluster_indices]
        
        vectorizer = TfidfVectorizer(max_features=num_words)
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)
        
        indices = tfidf_matrix.sum(axis=0).argsort()[0, ::-1]
        feature_names = vectorizer.get_feature_names_out()
        top_words = [feature_names[ind] for ind in indices[:num_words]]
        topic_to_words.append(top_words)
    
    # Create a dummy dictionary for compatibility
    dictionary = {i: word for i, word in enumerate(vectorizer.get_feature_names_out())}
    
    return model, dictionary, topic_to_words


# step2: nltk 依存句法分析


class DependencyParser():
    def __init__(self, model_path, parser_path):
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


def get_topic_sentiment_metrix(text, dictionary, lda_model, topic_word_metrix, dependency_parser, topic_nums=50):
    """获取主题-情感矩阵
    """
    text_p = word_segment(text)
    doc_bow = dictionary.doc2bow(text_p)  # 文档转换成bow
    doc_lda = lda_model[doc_bow]  # [(12, 0.042477883), (13, 0.36870235), (16, 0.35455772), (37, 0.20635633)]
    # print("doc_bow: ", doc_bow)
    # print("lda_model: ", doc_lda)
    # 初始化主题矩阵
    topci_sentiment_m = np.zeros(topic_nums)

    # 获取依存句法分析结果
    sentences = preprocessed(text)
    dep_parser_result_p = []
    for i in sentences:
        # 依存句法分析 phan tich cu phap
        print(i)
        dep_parser_result = dependency_parser.raw_parse(i)
        # print(dep_parser_result)
        for j in dep_parser_result:
            dep_parser_result_p.append([j[0][0], j[2][0]])
        print(dep_parser_result_p)
    print(doc_lda)
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