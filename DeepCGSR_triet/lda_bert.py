import numpy as np
from collections import Counter
from gensim.models.coherencemodel import CoherenceModel
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
from AutoEncoder import *
from preprocess_word import preprocess
import pandas as pd
import pickle
import os
# from datetime import datetime

class LDA_BERT:

    def save_model(self, filepath):
            """
            Save the trained model to a checkpoint file
            :param filepath: path to save the checkpoint file
            """
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
    
    def load_model(filepath):
        """
        Load the trained model from a checkpoint file
        :param filepath: path to the checkpoint file
        :return: the trained model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_topic_words(token_lists, labels, k=None):
        ''' Get topic within each topic form clustering results '''
        if k is None:
            k = len(np.unique(labels))
        topics = ['' for _ in range(k)]
        for i, c in enumerate(token_lists):
            topics[labels[i]] += (' ' + ' '.join(c))
        word_counts = list(map(lambda x: Counter(x.split()).items(), topics))
        # get sorted word counts
        word_counts = list(map(lambda x: sorted(x, key=lambda x: x[1], reverse=True),word_counts))
        # get topics
        topics = list(map(lambda x: list(map(lambda x: x[0], x[:10])), word_counts))
        return topics

    def get_coherence(model, token_lists, measure='c_v'):
        ''' Get model coherence from gensim.models.coherencemodel
        : param model: Topic_Model object
        : param token_lists: token list of docs
        : param topics: topics as top words 
        : param measure: coherence metrics
        : return: coherence score '''

        if model.method == 'LDA':
            cm = CoherenceModel(model=model.ldamodel, texts=token_lists, corpus = model.corpus, dictionary=model.dictionary, coherence = measure)
        else:
            model.topics = LDA_BERT.get_topic_words(token_lists, model.cluster_model.labels_)
            
            print("====================TOPICS=======================")
            print(model.topics)
            print("=================================================")
            cm = CoherenceModel(topics=model.topics, texts = token_lists, corpus=model.corpus, dictionary=model.dictionary, coherence = measure)
            return cm.get_coherence()
    
    def __init__(self, k=10, method='LDA_BERT', documents=None):
        """
        :param k: number of topics
        :param method: method chosen for the topic model
        """
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  # parameter for reletive importance of lda
        self.method = method
        self.AE = None
        self.data = None
        self.topics = None
        data = documents #pd.read_csv('/kaggle/working/train.csv')
        data = data.fillna('')  # only the comments has NaN's
        rws = data.reviewText
        self.token_lists = None
        sentences, self.token_lists, idx_in = preprocess(rws, samp_size=51000)
        self.fit(sentences, self.token_lists)

        print('Coherence:', self.get_coherence(self.token_lists, 'c_v'))
        print("===========================================================")

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vecotr representations from selected methods
        """

        if method == 'LDA':
            print('Getting vector representations for LDA ...')
            if not self.ldamodel:

                self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            print('Getting vector representations for LDA. Done!')
            return vec

        elif method == 'BERT':

            print('Getting vector representations for BERT ...')
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('bert-base-nli-max-tokens')
            vec = np.array(model.encode(sentences, show_progress_bar=True))
            print('Getting vector representations for BERT. Done!')
            return vec
        
        self.dictionary = corpora.Dictionary(token_lists)
        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
        vec_lda = self.vectorize(sentences, token_lists, method='LDA')
        vec_bert = self.vectorize(sentences, token_lists, method='BERT')
        vec_ldabert = np.c_[vec_lda * self.gamma, vec_bert]
        self.vec['LDA_BERT_FULL'] = vec_ldabert
        if not self.AE:
            self.AE = Autoencoder()
            print('Fitting Autoencoder ...')
            self.AE.fit(vec_ldabert)
            print('Fitting Autoencoder Done!')
        vec = self.AE.encoder.predict(vec_ldabert)
        return vec

    def fit(self, sentences, token_lists, method=None, m_clustering=None):
        """
        Fit the topic model for selected method given the preprocessed data
        :docs: list of documents, each doc is preprocessed as tokens
        :return:
        """
        # Default method
        if method is None:
            method = self.method
        # Default clustering method
        if m_clustering is None:
            m_clustering = KMeans

        # turn tokenized documents into a id <-> term dictionary
        if not self.dictionary:
            self.dictionary = corpora.Dictionary(token_lists)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
        
        print('Clustering embeddings ...')
        self.cluster_model = m_clustering(self.k)
        self.vec[method] = self.vectorize(sentences, token_lists, method)
        self.cluster_model.fit(self.vec[method])
        print('Clustering embeddings. Done!')

if __name__ == '__main__':
    path = "./data/All_Beauty_5.json"
    data = pd.read_json(path, lines=True)

    count = 0
    index = []
    for i in range(len(data)):
    #print(i)
        if type(data.iloc[i, 7]) == float:
            count += 1
        else:
            index.append(i)

    documents = data.iloc[index, 7]
    documents = documents.reset_index()
    documents.drop('index', inplace = True, axis = 1)

    # create data frame with all abstracts, use as input corpus
    documents['index'] = documents.index.values
    documents.head()
    method = "LDA_BERT"
    samp_size = 51000
    ntopic = 10
    checkpoint_path = "chkpt/model_TBERT_checkpoint.pkl"
    if os.path.exists(checkpoint_path):
        # Load the model from the checkpoint
        tm = LDA_BERT.load_model(checkpoint_path)
    else:
        # Train the model
        tm = LDA_BERT(k=ntopic, method=method, documents=documents)
        # Save the trained model as a checkpoint
        tm.save_model(checkpoint_path)
    topic_to_words = []
    print(tm.get_topic_words(tm.token_lists, tm.cluster_model.labels_))
    # for i in range(ntopic):
    #     cur_topic_words = [ele[0] for ele in tm.show_topic(i, ntopic)]
    #     topic_to_words.append(cur_topic_words)
    # print(topic_to_words)


    

    
