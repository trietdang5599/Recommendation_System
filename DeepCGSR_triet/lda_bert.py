import numpy as np
from collections import Counter
from gensim.models.coherencemodel import CoherenceModel
from sklearn.cluster import KMeans
from gensim import corpora
import gensim
from AutoEncoder import *
from preprocess_word import preprocess
import pandas as pd
# from datetime import datetime

class LDA_BERT:

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
            topics = LDA_BERT.get_topic_words(token_lists, model.cluster_model.labels_)
            cm = CoherenceModel(topics=topics, texts = token_lists, corpus=model.corpus, dictionary=model.dictionary, coherence = measure)
            return cm.get_coherence()
    
    def __init__(self, k=10, method='LDA_BERT'):
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
        # self.id = method + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

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

    # def predict(self, sentences, token_lists, out_of_sample=None):
    #     """
    #     Predict topics for new_documents
    #     """
    #     # Default as False
    #     out_of_sample = out_of_sample is not None

    #     if out_of_sample:
    #         corpus = [self.dictionary.doc2bow(text) for text in token_lists]
    #         if self.method != 'LDA':
    #             vec = self.vectorize(sentences, token_lists)
    #             print(vec)
    #     else:
    #         corpus = self.corpus
    #         vec = self.vec.get(self.method, None)

    #     if self.method == "LDA":
    #         lbs = np.array(list(map(lambda x: sorted(self.ldamodel.get_document_topics(x),
    #                                                  key=lambda x: x[1], reverse=True)[0][0],
    #                                 corpus)))
    #     else:
    #         lbs = self.cluster_model.predict(vec)
    #     return lbs
if __name__ == '__main__':
    path = "./data/All_Beauty_5.json"
    meta = pd.read_json(path, lines=True)

    count = 0
    index = []
    for i in range(len(meta)):
    #print(i)
        if type(meta.iloc[i, 7]) == float:
            count += 1
        else:
            index.append(i)

    print(len(index), 'Paper have abstract available')
    documents = meta.iloc[index, 7]
    documents = documents.reset_index()
    documents.drop('index', inplace = True, axis = 1)

    # create data frame with all abstracts, use as input corpus
    documents['index'] = documents.index.values
    documents.head()
    method = "LDA_BERT"
    samp_size = 51000
    ntopic = 10
    #def model(): #:if __name__ == '__main__':
    tm = LDA_BERT(k = ntopic, method = method)

    data = documents #pd.read_csv('/kaggle/working/train.csv')
    data = data.fillna('')  # only the comments has NaN's
    rws = data.reviewText
    sentences, token_lists, idx_in = preprocess(rws, samp_size=samp_size)

    tm.fit(sentences, token_lists)
    print('Coherence:', tm.get_coherence(token_lists, 'c_v'))
    print("===========================================================")