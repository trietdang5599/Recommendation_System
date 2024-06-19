import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset
from torchfm.model.fm import FactorizationMachineModel
from config import args

def TransformLabel(data, user_encoder, item_encoder, selected_fields, csv_path):
  user = LabelEncoder()
  user.fit(user_encoder)
  data[:, 0] = user.transform(user_encoder)
  item = LabelEncoder()
  item.fit(item_encoder)
  data[:, 1] = item.transform(item_encoder)
  
  with open(csv_path, 'w', newline='') as csv_data:
    csv_writer = csv.writer(csv_data)

    # Ghi tiêu đề
    csv_writer.writerow(selected_fields)

    # Ghi dữ liệu từ tệp JSON vào tệp CSV
    for item in data:
        csv_writer.writerow(item)

def DecodeLabel(encoded_data, user_encoder, item_encoder):
    decoded_data = encoded_data.copy()
    decoded_data['reviewerID'] = user_encoder.inverse_transform(encoded_data['reviewerID'])
    decoded_data['itemID'] = item_encoder.inverse_transform(encoded_data['itemID'])
    return decoded_data


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

    with open(csv_path, 'w', newline='') as csv_data:
        
        csv_writer = csv.writer(csv_data)

        # Ghi tiêu đề
        csv_writer.writerow(selected_fields)

        # Ghi dữ liệu từ data vào tệp CSV
        csv_writer.writerows(data)

class ReviewAmazon():
    
  def __init__(self, data_path, dataset_path, sep=',', engine='c', header='infer'):
    data = pd.read_csv(data_path, sep=sep, engine=engine, header=header,
                       usecols=['reviewerID', 'itemID', 'overall']).to_numpy()[:, :3]
    print("====================================")
    TransformLabel(data, data[:, 0], data[:, 1], ['reviewerID', 'itemID', 'overall'], dataset_path)
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
  
def get_dataset(name, path):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'reviewAmazon':
        return ReviewAmazon(data_path=args.data_path,dataset_path=path)

def get_model(dataset):
    field_dims = dataset.field_dims
    print("dataset_shape: ", len(dataset))
    return FactorizationMachineModel(field_dims, embed_dim=20)