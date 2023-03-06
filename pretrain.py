from model.backbones import ImageModelWarpper
import torch
from data import data_provider
import numpy as np
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import os
import time
import argparse
from utils.tools import EarlyStopping, OptimWrapper


fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
parser = argparse.ArgumentParser(description='God please help me')

# basic config
parser.add_argument('--pr_model', type=str, default='gc_efficientnetv2_rw_t')
parser.add_argument('--resume', action='store_true', default=False)
parser.add_argument('--proj_method', type=str, default="linear")
parser.add_argument('--d_features', type=int, default=1024)
parser.add_argument('--freqm', type=int, default=24)
parser.add_argument('--timem', type=int, default=24)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--nCategories', type=int, default=36)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--warmup', type=int, default=3000)
parser.add_argument('--lr_min', type=float, default=1e-4)
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--patience', type=int, default=50)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--clip_samples', type=int, default=16000)
parser.add_argument('--sampling_rate', type=int, default=16000)
parser.add_argument('--audio_dataset', type=str, default='scv2')

args = parser.parse_args()
args.c_in, args.num_classes = 1, 35

model = ImageModelWarpper(args)

_, train_loader = data_provider(args, 'train', 'audio')
_, test_loader = data_provider(args, 'test', 'audio')

optimizer = optimizer=optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95, verbose=True)
criterion = nn.CrossEntropyLoss(reduction='mean')
device = torch.device('cuda:0')
model.to(device)
setting = f"{args.pr_model}_ds{args.audio_dataset}"

def vali():
    model.eval()
    total_loss = []
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_label in test_loader:
            batch_x = batch_x.float().to(device)
            batch_label = batch_label.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_label)
            total_loss.append(loss.item())
            pred_class = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
            preds.append(pred_class.detach().cpu().numpy())
            trues.append(batch_label.cpu().numpy())
    total_loss = np.average(total_loss)
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    acc = np.sum(trues == preds) / len(trues)
    model.train()
    return total_loss, acc


def train():
    path = os.path.join('./pretrained/', setting)
    if not os.path.exists(path):
        os.makedirs(path)
        best_model_path = f"{path}/checkpoint.pth"
        model.load_state_dict(torch.load(best_model_path))
        
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    for epoch in range(args.train_epochs):
        epoch_time = time.time()

        train_loss = []
        model.train()
        for i, (batch_x, batch_label) in enumerate(train_loader): # [B, D, L]
            batch_label = batch_label.to(device)
            batch_x = batch_x.float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            ave_loss = np.average(train_loss)

            if (i+1) % 100==0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {'%.7f'%ave_loss}") 
        scheduler.step()
        test_loss, acc = vali()

        print(f"Epoch: {epoch + 1}, cost time: {time.time()-epoch_time} | Test Loss: {'%.7f'%test_loss} Test acc: {'%.4f'%acc}")
        early_stopping(acc, {args.pr_model: model}, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    best_model_path = f"{path}/checkpoint.pth"
    model.load_state_dict(torch.load(best_model_path))
    

def test():
    model.load_state_dict(torch.load(f"./pretrained/{setting}/checkpoint.pth"))
    model.eval()
    
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_label in test_loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)
            pred_class = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
            preds.append(pred_class.detach().cpu().numpy())
            trues.append(batch_label.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    acc = np.sum(trues == preds) / len(trues)
    
    # result save
    folder_path = './results/' + setting +'/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    np.save(folder_path + 'preds.npy', preds)
    np.save(folder_path + 'trues.npy', trues)
    print("test accuracy: {0}".format(acc))

def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %40s %12s' % ('layer', 'name', 'parameters'))
    param_list = list(model.named_parameters())
    for i, (name, p) in enumerate(param_list):
        name = name.replace('module_list.', '')
        print('%5g %40s %12g' % (i, name, p.numel()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))
 

if __name__ == '__main__':
    # model_info(model)
    train()
    test()