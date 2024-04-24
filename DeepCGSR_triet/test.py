import numpy as np
from scipy.sparse import csc_matrix

class RecommendationModel:
    def __init__(self, data, len_user=3, len_item=4):
        self.data = data
        self.len_user = len_user
        self.len_item = len_item
        self.sparse_ratings = self.create_sparse_matrix(data, len_user, len_item)
        self.rows, self.cols, _ = zip(*data)  # Khởi tạo self.rows và self.cols

    def create_sparse_matrix(self, data, len_user, len_item):
        rows, cols, vals = zip(*data)
        return csc_matrix((vals, (rows, cols)), shape=(len_user, len_item))

    def predict(self, emb_user, emb_item):
        p_ratings = np.dot(emb_user, emb_item.transpose())
        return p_ratings

    def cost(self, emb_user, emb_item):
        p_predict = self.predict(emb_user, emb_item)
        p_data = [p_predict[r][c] for r, c in zip(self.rows, self.cols)]
        predicted = self.create_sparse_matrix(list(zip(self.rows, self.cols, p_data)), self.len_user, self.len_item)
        return np.sum((self.sparse_ratings - predicted).power(2)) / len(self.data) # lay ma tran ban dau - ma tran thua, binh phuong, cong lai, chia do dai

# Tạo một instance của lớp RecommendationModel với dữ liệu giả định
data = [(0, 0, 4), (1, 1, 3), (2, 2, 5), (0, 3, 2), (2, 3, 1)]
model = RecommendationModel(data)

# Tạo các vectơ biểu diễn người dùng và sản phẩm
emb_user = np.random.rand(3, 4)  # 3 người dùng, mỗi người có biểu diễn 4 chiều
emb_item = np.random.rand(4, 4)  # 4 sản phẩm, mỗi sản phẩm có biểu diễn 4 chiều

# In ra giá trị chi phí
cost_value = model.cost(emb_user, emb_item)
print("Giá trị chi phí:", cost_value)
