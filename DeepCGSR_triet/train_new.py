import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss
import csv
import ast
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture
class FullyConnectedModel(nn.Module):
    def __init__(self, num_features, output_dim=1):
        super(FullyConnectedModel, self).__init__()
        self.fc = nn.Linear(num_features, output_dim, bias=False)
        self.user_bias = nn.Parameter(torch.zeros(num_users))
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, user_features, item_features, user_indices, item_indices):
        # Element-wise multiplication of user and item features
        user_features = torch.tensor(user_features)
        # print(user_features.shape)
        item_features = torch.tensor(item_features)
        interaction = user_features * item_features
        # Sum over features
        # print(interaction.shape)
        interaction = interaction.sum(dim=1)
        # print(interaction.shape)
        # Multiply by weights
        prediction = self.fc(interaction.to(dtype=torch.float32))
        # Add biases
        prediction += self.global_bias + self.user_bias[user_indices] + self.item_bias[item_indices]
        return prediction.squeeze()

# Instantiate the model
num_users, num_items = 990, 85  # Example numbers for users and items
num_features = 20  # This should match the size of your U_deep and I_deep feature vectors
model = FullyConnectedModel(num_features)

def load_data(csv_file):
    data = []
    with open(csv_file, 'r', newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    return data

dataset = load_data('data/ratings_AB.csv')
train_loader = DataLoader(dataset, batch_size=20, shuffle=True)

# Initialize the model
model = FullyConnectedModel(num_features, output_dim=20)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 2
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
    # Xử lý batch ở đây
        user_idx = [int(x) for x in batch['reviewerID']]
        item_idx = [int(x) for x in batch['asin']]
        rating = [float(x) for x in batch['overall']]
        list_udeep = []
        for item in batch['Udeep']:
            list_udeep.append(np.array(ast.literal_eval(item)))
        user_feature = list_udeep
        list_ideep = []
        for item in batch['Ideep']:
            list_ideep.append(np.array(ast.literal_eval(item)))
        item_feature = list_ideep
        
        # Forward pass
        predictions = model(user_feature, item_feature, user_idx, item_idx)
        
        # Calculate loss
        loss = criterion(predictions, torch.tensor(rating))  # rating needs to be a column vector
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")