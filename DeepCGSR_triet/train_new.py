import numpy as np
import tqdm
from config import args
from DeepCGSR import merge_csv_columns
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss
import csv
import ast
import torch.nn as nn
import torch.optim as optim

class EarlyStopper(object):

    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path) # best model
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
# Define the model architecture
class FullyConnectedModel(nn.Module):
    def __init__(self, batch_size, output_dim=1):
        super(FullyConnectedModel, self).__init__()
        self.fc = nn.Linear(batch_size, output_dim, bias=False)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.global_bias = nn.Parameter(torch.zeros(1))
        # self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices, user_features, item_features):
        # Element-wise multiplication of user and item features
        user_features = torch.tensor(user_features)
        # print(user_features.shape)
        item_features = torch.tensor(item_features)
        interaction = user_features * item_features
        # Sum over features
        
        interaction = interaction.sum(dim=1)
        # print(interaction.shape)
        self.fc = nn.Linear(len(interaction), len(interaction), bias=True)
        # Multiply by weights
        prediction = self.fc(interaction.to(dtype=torch.float32))
        # Add biases
        prediction += self.global_bias + self.user_bias[user_indices] + self.item_bias[item_indices]
        # prediction = self.sigmoid(prediction.squeeze(0))
        
        return prediction.squeeze()

def reprocess_input(data):
    user_idx = [int(x) for x in data['reviewerID']]
    item_idx = [int(x) for x in data['asin']]
    rating = [float(x) for x in data['overall']]
    list_udeep = []
    for item in data['Udeep']:
        list_udeep.append(np.array(ast.literal_eval(item)))
    user_feature = list_udeep
    list_ideep = []
    for item in data['Ideep']:
        list_ideep.append(np.array(ast.literal_eval(item)))
    item_feature = list_ideep
    return user_idx, item_idx, rating, user_feature, item_feature


# def __preprocess_target(target):
#     target[target <= 3] = 0
#     target[target > 3] = 1
#     return target.astype(np.float32)

def load_data(csv_file):
    merge_csv_columns('data/ratings_AB.csv', 'reviewerID', 'transformed_udeep.csv', 'ID', 'Array', 'Udeep')
    merge_csv_columns('data/ratings_AB.csv', 'asin', 'transformed_ideep.csv', 'ID', 'Array', 'Ideep')
    count = 0
    data = []
    with open(csv_file, 'r', newline='') as file:
        
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            # if(row['overall'] == '5.0'):
            #     row['overall'] = '2.0'
            data.append(row)
        # for row in data:
        #     if(float(row['overall']) < 4):
        #         count = count + 1
    # overall = np.array([float(row['overall']) for row in data])
    # overall = __preprocess_target(overall)
    # for i in range(len(data)):
    #     data[i]['overall'] = overall[i]
    # print("Item <= 3: ", count)
    return data

def calculate_rmse(y_true, y_pred):
    """
    Tính toán Root Mean Square Error (RMSE) giữa y_true và y_pred.
    
    Tham số:
    - y_true: mảng numpy chứa các giá trị thực tế.
    - y_pred: mảng numpy chứa các giá trị dự đoán.
    
    Trả về:
    - Giá trị RMSE.
    """
    # Chuyển đổi danh sách thành mảng NumPy
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    # Tính toán squared_errors
    squared_errors = (y_true_np - y_pred_np) ** 2
    # Tính trung bình của bình phương sai số
    mean_squared_error = np.mean(squared_errors)
    # Tính RMSE bằng cách lấy căn bậc hai của trung bình bình phương sai số
    rmse = np.sqrt(mean_squared_error)
    return rmse

def test_rsme(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for data in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            user_idx, item_idx, target, udeep, ideep = reprocess_input(data)
            y = model(user_idx, item_idx, udeep, ideep)
            targets.extend(target)
            # print("Max: ", max(y))
            # print("Min: ", min(y))
            predicts.extend([float(pred) for pred in y.flatten().cpu().numpy()])

    
    # return roc_auc_score(targets, predicts) 
    return calculate_rmse(targets, predicts)

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for data in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            user_idx, item_idx, target, udeep, ideep = reprocess_input(data)
            y = model(user_idx, item_idx, udeep, ideep)
            targets.extend(target)
            predicts.extend([round(float(pred)) for pred in y.flatten().cpu().numpy()])

    accuracy = accuracy_score(targets, predicts)
    return accuracy

def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    # Train the model
    model.train()
    total_loss = 0

    for batch in data_loader:
        try:
        # Xử lý batch ở đây
            user_idx, item_idx, rating, user_feature, item_feature = reprocess_input(batch)
            
            # Forward pass
            predictions = model(user_idx, item_idx, user_feature, item_feature)

            # Calculate loss
            loss = criterion(predictions, torch.tensor(rating))  # rating needs to be a column vector
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        except Exception as e:
            # Code block to handle the exception if it occurs
            # This block will be executed if an exception of type ExceptionType is raised
            print("error: ",e)
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}")
        
        
# Instantiate the model
num_users, num_items = 991, 85  # Example numbers for users and items
num_features = 20  # This should match the size of your U_deep and I_deep feature vectors
batch_size = 32
epoch = 10
device='cuda:0'



dataset = load_data('data/ratings_AB.csv')
# train_loader = DataLoader(dataset, batch_size, shuffle=True)
train_length = int(len(dataset) * 0.7)
valid_length = int(len(dataset) * 0.1)
test_length = len(dataset) - train_length - valid_length
print(train_length, valid_length, test_length)
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8)
# Initialize the model
model = FullyConnectedModel(batch_size, output_dim=20)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
filename = 'triet_deepcgsr'
early_stopper = EarlyStopper(num_trials=3, save_path=f'{args.save_dir}/{filename}.pt')
rsme = 0
count = 0
for epoch_i in range(epoch):
    train(model, optimizer, train_data_loader, criterion, device)
    auc = test(model, valid_data_loader, device)
    print('epoch:', epoch_i, 'validation: auc:', auc)
    rsme = rsme + test_rsme(model, valid_data_loader, device)
    print('epoch:', epoch_i, 'validation: rsme:', rsme)
    count = count + 1
    if not early_stopper.is_continuable(model, auc):
        print(f'validation: best auc: {early_stopper.best_accuracy}')
        
        break
auc = test(model, test_data_loader, device)
print(f'test auc: {auc}')
# rsme = test_rsme(model, test_data_loader, device)
# print(f'test rsme:', {rsme})
print(f'test rsme:', {rsme/count})

