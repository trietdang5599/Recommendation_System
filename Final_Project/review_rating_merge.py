import svd
from svd import SVD
import csv
import numpy as np
import pandas as pd
import ast
import torch
import os
from fine_coarse_merge import GetReviewFeatures, GetReview_ItemFeatures, CreateAndWriteCSV
from config import args
from torch.utils.data import Dataset, DataLoader, Subset
from data_process import TransformLabel, DecodeLabel
from sklearn.preprocessing import LabelEncoder

checkpoint_path = 'chkpt/svd.pt'
allFeatureReview = GetReviewFeatures()
users = allFeatureReview['reviewerID'].tolist()
items = allFeatureReview['itemID'].tolist()

allFeatureReview_translabel = allFeatureReview.copy()

allFeatureReview_translabel['Original_ReviwerID'] = pd.Series(users, index=allFeatureReview.index)
allFeatureReview_translabel['Original_ItemID'] = pd.Series(items, index=allFeatureReview.index)
print(allFeatureReview_translabel)
allFeatureReview_translabel.to_csv('feature/allFeatureReview_translabel.csv', index=False)

TransformLabel(allFeatureReview_translabel.to_numpy(), users, items, ['reviewerID', 'itemID', 'overall', 'fine_feature', 'coarse_feature', 'Original_ReviwerID', 'Original_ItemID'], 'feature/allFeatureReview_translabel.csv')
allFeatureReview_translabel = pd.read_csv('feature/allFeatureReview_translabel.csv', sep=',', engine='c', header='infer')
print(allFeatureReview_translabel)

# Tính toán độ dài của từng tập dữ liệu
total_length = len(allFeatureReview_translabel)
train_length = int(total_length * 0.7)
valid_length = int(total_length * 0.1)
test_length = total_length - train_length - valid_length

# Tạo chỉ mục ngẫu nhiên cho việc chia dữ liệu
indices = np.random.permutation(total_length)

# Chia dữ liệu thành các tập huấn luyện, xác thực và kiểm tra
train_indices = indices[:train_length]
valid_indices = indices[train_length:train_length+valid_length]
test_indices = indices[train_length+valid_length:]

# Tạo DataFrame cho từng tập dữ liệu
train_data_df = allFeatureReview_translabel.iloc[train_indices]
valid_data_df = allFeatureReview_translabel.iloc[valid_indices]
test_data_df = allFeatureReview_translabel.iloc[test_indices]

# Kiểm tra độ dài của từng tập dữ liệu
print("Độ dài của tập huấn luyện:", len(train_data_df))
print("Độ dài của tập xác thực:", len(valid_data_df))
print("Độ dài của tập kiểm tra:", len(test_data_df))
# train_data_df = DecodeLabel(train_data_df, user_encoder, item_encoder)
# print(train_data_df)
train_data_df['reviewerID'] = train_data_df['Original_ReviwerID']
train_data_df['itemID'] = train_data_df['Original_ItemID']
train_data_df = train_data_df.drop(columns=['Original_ReviwerID', 'Original_ItemID'])
reviewer_feature_dict, item_feature_dict = GetReview_ItemFeatures(train_data_df)
args.user_length = len(reviewer_feature_dict)
args.item_length = len(item_feature_dict)

train_data_df.to_csv('feature/allFeatureReview_train.csv', index=False)
valid_data_df.to_csv('feature/valid_data_df.csv', index=False)
test_data_df.to_csv('feature/test_data_df.csv', index=False)

# Kiểm tra xem tệp checkpoint có tồn tại không
if os.path.exists(checkpoint_path):
    # Tải mô hình từ checkpoint
    svd = torch.load(checkpoint_path)
    
else:
    svd = SVD('feature/allFeatureReview_train.csv')
    svd.train() 
    torch.save(svd, checkpoint_path)

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