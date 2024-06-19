import os
import pandas as pd
import gzip
import json
import numpy as np
import scipy.sparse as sparse
import torch

class SVD():
    def __init__(self, data_path):

        self.df = pd.read_csv(data_path)
        self.users = sorted(self.df['reviewerID'].unique()) # group by reviewerID --> count users
        self.items = sorted(self.df['itemID'].unique()) # group by asin --> count items

        self.users_id_dict = {u:index for index,u in enumerate(self.users)}
        self.items_id_dict = {i:index for index,i in enumerate(self.items)}

        self.rows = []
        self.cols = []
        self.data = []

        self.beta = 0.9
        self.lmbda = 0.0002
        self.k = 10 # why?
        self.learning_rate = 0.01
        self.iterations = 1000
        self.u_dim = len(self.users) #991
        self.i_dim = len(self.items) #85

        self.init_ratings_matric()


    def zip_loader(self):
        g = gzip.open(self.data_path, 'rb')
        for l in g:
            yield json.loads(l)

    def getDF(self):
        i = 0
        df = {}
        for d in self.zip_loader():
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def init_ratings_matric(self):
        for item in self.df.itertuples():
            r = item[3]
            u = item[1]
            i = item[2]
            iu = self.users.index(u)
            ii = self.items.index(i)
            self.rows.append(iu)
            self.cols.append(ii)
            self.data.append(r)
        ratings = np.zeros((len(self.users), len(self.items)))

        for r, c, d in zip(self.rows, self.cols, self.data):
            ratings[int(r), int(c)] = d

        self.ratings = ratings
        self.sparse_ratings = self.create_sparse_matrix(self.data,self.u_dim,self.i_dim)


    def create_sparse_matrix(self,data,  len_user, len_item):
        return sparse.csc_matrix((data, (self.rows, self.cols)), shape=(len_user, len_item)) # tạo ma trận thưa

    def create_embeddings(self,n):
        return 6 * np.random.random((n, self.k)) / self.k

    def predict(self,emb_user, emb_item):
        p_ratings = np.dot(emb_user, emb_item.transpose()) # multiple 2 matrix (user & item) transpose to make 2 matrix has the same dim
        return p_ratings

    def cost(self, emb_user, emb_item):
        p_predict = self.predict(emb_user, emb_item)
        p_data = [p_predict[r][c] for r, c in zip(self.rows, self.cols)]
        predicted = self.create_sparse_matrix(p_data, emb_user.shape[0], emb_item.shape[0]) # tạo ma trận thưa bằng với kích cỡ row
        return np.sum((self.sparse_ratings - predicted).power(2)) / len(self.data) # lấy ma trận ban đầu - ma trận thưa mới tạo được rồi bình phương, 
                                                                                   # cộng lại và chia cho độ dài ma trận ban đầu

    def gradient(self,  emb_user, emb_item): # tốc độ thay đổi của hàm số bằng đạo hàm các biến 
        p_predict = self.predict(emb_user, emb_item) 
        p_data = [p_predict[r][c] for r, c in zip(self.rows, self.cols)] #rating
        sparse_predicted = self.create_sparse_matrix(p_data,  emb_user.shape[0], emb_item.shape[0])
        delta = (self.sparse_ratings - sparse_predicted) # chênh lệch giữa ma trận thưa thực tế và dự đoán.
        grad_user = (-2 / self.df.shape[0]) * (delta * emb_item) + 2 * self.lmbda * emb_user # Tính đạo hàm riêng của hàm chi phí theo người dùng
        grad_item = (-2 / self.df.shape[0]) * (delta.T * emb_user) + 2 * self.lmbda * emb_item # Tính đạo hàm riêng của hàm chi phí theo sản phẩm
        # print("delta: ", delta, "grad_user: ", grad_user, "grad_item: ", grad_item)
        return grad_user, grad_item # => tìm ra hướng thay đổi mạnh mẽ nhất của hàm , để tìm ra cực tiểu và cực đại địa phương

    def train(self):
        emb_user = self.create_embeddings(self.u_dim)
        # print("User:",emb_user)
        emb_item = self.create_embeddings(self.i_dim)
        # print("Item", emb_item)
        grad_user, grad_item = self.gradient(emb_user, emb_item)
        v_user = grad_user
        v_item = grad_item
        for i in range(self.iterations):
            grad_user, grad_item = self.gradient(emb_user, emb_item)
            v_user = self.beta * v_user + (1 - self.beta) * grad_user 
            v_item = self.beta * v_item + (1 - self.beta) * grad_item
            emb_user = emb_user - self.learning_rate * v_user
            emb_item = emb_item - self.learning_rate * v_item
            if (not (i + 1) % 50):
                print("\niteration", i + 1, ":")
                print("train mse:", np.sqrt(self.cost(emb_user, emb_item))) # RMSE thêm căn bậc 2
        self.emb_user = emb_user
        self.emb_item = emb_item

    def get_embedings(self):
        if self.emb_user is not None:
            return self.emb_user,self.emb_item
        else:
            raise Exception(print('Please train the model first.'))

    def get_user_embedding(self,user_id):
        index = self.users_id_dict[user_id]
        return self.emb_user[index,:]

    def get_item_embedding(self,item_id):
        index = self.items_id_dict[item_id]
        return self.emb_item[index,:]



if __name__ == '__main__':
    svd = SVD()
    svd.train()
    emb_user,emb_item = svd.get_embedings()
    print(emb_user.shape)
    print(svd.get_user_embedding('A3CIUOJXQ5VDQ2'))



