from dataloader import MovieLens20MDataset
import model
from model import FactorizationMachine
from model import FactorizationMachineModel
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import torch
import tqdm

path = '/home/jwjung/Recsys/ml-20m/ml-20m/ratings.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ml = MovieLens20MDataset(path)

train_length = int(len(ml) * 0.7)
valid_length = int(len(ml) * 0.1)
test_length = len(ml) - train_length - valid_length
train, valid, test = torch.utils.data.random_split(ml, (train_length, valid_length ,test_length))

batch_size = 1024

train_data_loader = DataLoader(train, batch_size= batch_size, num_workers = 8)
valid_data_loader = DataLoader(valid, batch_size= batch_size, num_workers = 8)
test_data_loader = DataLoader(test, batch_size= batch_size, num_workers = 8)


def train(model, optimizer, data_loader, criterion, device, log_interval = 100):

    model.train()
    total_loss = 0

    
    tk0 = tqdm.tqdm(train_data_loader, smoothing=0, mininterval = 1.0)
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)
        
model = FactorizationMachineModel(ml.field_dims, embed_dim= 16).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001, weight_decay= 1e-6)

epochs = 100

for epoch in range(epochs):
    print(f"epoch: {epoch} / {epochs}")
    train(model = model,optimizer = optimizer, data_loader= train_data_loader, criterion=criterion, device = device)
    auc = test(model, valid_data_loader, device)
    print(f'epoch: {epoch},  validation: : {auc}')
 
auc = test(model, test_data_loader, device)
print(f'test auc: {auc}')








        
