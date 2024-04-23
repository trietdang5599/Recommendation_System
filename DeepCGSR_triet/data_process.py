import pandas as pd
import gzip
import json
import csv
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

# df = getDF('data/All_Beauty_5.json.gz')
df = getDF('data/All_Beauty_Filtered.json.gz')
# print(df.head())
# print(df.iloc[0])

def TransformLabel(data, csv_path):
  user = LabelEncoder()
  user.fit(data[:, 0])
  data[:, 0] = user.transform(data[:, 0])
  item = LabelEncoder()
  item.fit(data[:, 1])
  data[:, 1] = item.transform(data[:, 1])
  selected_fields = ['reviewerID', 'itemID', 'overall']
  with open(csv_path, 'w', newline='') as csv_data:
    csv_writer = csv.writer(csv_data)

    # Ghi tiêu đề
    csv_writer.writerow(selected_fields)

    # Ghi dữ liệu từ tệp JSON vào tệp CSV
    for item in data:
        csv_writer.writerow(item)


def TransformLabel_Deep(data, csv_path):
     # Khởi tạo LabelEncoder cho user
    user_encoder = LabelEncoder()

    # Lấy các giá trị reviewerID từ cột đầu tiên của data
    reviewer_ids = data[:, 0]

    # Fit và transform user
    user_encoded = user_encoder.fit_transform(reviewer_ids)

    # Thay thế cột reviewerID trong data bằng các giá trị đã được mã hóa
    data[:, 0] = user_encoded

    # Chọn các trường cần ghi vào CSV
    selected_fields = ['ID', 'Array']

    # with open(csv_path, 'r', newline='') as csv_file:
    #     csv_reader = csv.DictReader(csv_file)
    #     for row in csv_reader:
    #       print(row["Array"])
    with open(csv_path, 'w', newline='') as csv_data:
        
        csv_writer = csv.writer(csv_data)

        # Ghi tiêu đề
        csv_writer.writerow(selected_fields)

        # Ghi dữ liệu từ data vào tệp CSV
        csv_writer.writerows(data)

class ReviewAmazon():
    
  def __init__(self, data_path, dataset_path, sep=',', engine='c', header='infer'):
    # SeperateData(dataset_path)
    # data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
    data = pd.read_csv(data_path, sep=sep, engine=engine, header=header,
                       usecols=['reviewerID', 'itemID', 'overall']).to_numpy()[:, :3]
    # data = pd.read_csv("./feature/allFeatureReview.csv", sep=sep, engine=engine, header=header,
    #                    usecols=['reviewerID', 'itemID', 'overall']).to_numpy()[:, :3]
    
    # data.columns = ['ReviewerID', 'ItemID', 'overall']
    # print(data.head())
    print("====================================")
    # if os.path.exists(dataset_path) == False:
    TransformLabel(data, dataset_path)
    self.items = data[:, :2].astype(np.int)  # -1 because ID begins from 1
    self.targets = self.__preprocess_target(data[:, 2]).astype(np.float32)
    self.field_dims = np.max(self.items, axis=0) + 1
    self.user_field_idx = np.array((0, ), dtype=np.long)
    self.item_field_idx = np.array((1,), dtype=np.long)

  def __len__(self):
    return self.targets.shape[0]

  def __getitem__(self, index):
      return self.items[index], self.targets[index]

  def __preprocess_target(self, target):
      target[target <= 3] = 0
      target[target > 3] = 1
      return target
  
