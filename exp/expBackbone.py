from data import data_provider
from exp.expBasic import Basic
from model.backbones import MyModelWarpper
from sklearn.manifold import TSNE
from utils.tools import EarlyStopping, OptimWrapper
from utils.utils import mkdir
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import time

import warnings
warnings.filterwarnings('ignore')

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer, AvgNonZeroReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses, miners
        
    
class BackboneExp(Basic):
    def __init__(self, args):
        super().__init__(args)
        self.model = self._build_model().to(self.device)
        _, self.train_loader = data_provider(args, 'train', 'ts')
        _, self.test_loader = data_provider(args, 'test', 'ts')
        self.miner = miners.MultiSimilarityMiner()
        self.d_criterion = losses.SupConLoss(
                                distance = CosineSimilarity(), 
                                reducer = AvgNonZeroReducer(), 
                                embedding_regularizer = LpRegularizer()
                                )

        self.criterion = nn.CrossEntropyLoss(reduce='mean')
        self.optim_g, self.optim_f, self.scheduler = self._select_optimizer()
    
    def _build_model(self):
        model = MyModelWarpper(self.args)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        optim_g = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.model.parameters()), 
            lr=self.args.learning_rate)
        optim_f = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optim_f, gamma=0.9, verbose=True)
        return optim_g, optim_f, scheduler

    def vali(self):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        with torch.no_grad():
            for batch_x, batch_label in self.test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.to(self.device)
                outputs, _ = self.model(batch_x)
                loss = self.criterion(outputs, batch_label)
                total_loss.append(loss.item())
                pred_class = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
                preds.append(pred_class.detach().cpu().numpy())
                trues.append(batch_label.cpu().numpy())
        total_loss = np.average(total_loss)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        acc = np.sum(trues == preds) / len(trues)
        self.model.train()
        return total_loss, acc
    

    def train(self):
        path = os.path.join(self.args.checkpoint_path, self.args.setting)
        backbone_path = os.path.join(self.args.backbone_path, self.args.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(backbone_path):
            os.makedirs(backbone_path)
        if self.args.resume:
            best_model_path = f"{path}/checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))
            
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        
        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()

            train_f_loss = []
            train_g_loss = []
            self.model.train()
            for i, (batch_x, batch_label) in enumerate(self.train_loader): # [B, D, L]
                batch_label = batch_label.to(self.device)
                batch_x = batch_x.float().to(self.device)
                self.optim_g.zero_grad()
                self.optim_f.zero_grad()
                outputs, embeddings = self.model(batch_x)
                if self.args.use_metric:
                    f_loss = self.criterion(outputs, batch_label)
                    f_loss.backward()
                    self.optim_f.step()
                    
                    outputs, embeddings = self.model(batch_x)
                    hard_pairs = self.miner(embeddings, batch_label)
                    g_loss = self.d_criterion(embeddings, batch_label, hard_pairs)
                    g_loss.backward()
                    self.optim_g.step()
                    
                    train_f_loss.append(f_loss.item())
                    ave_f_loss = np.average(train_f_loss)
                    
                    train_g_loss.append(g_loss.item())
                    ave_g_loss = np.average(train_g_loss)

                    if (i+1) % 20==0:
                        print(f"\titers: {i + 1}, epoch: {epoch + 1} | CN loss: {'%.7f'%ave_f_loss} Metric loss: {'%.7f'%ave_g_loss}") 
                else:
                    f_loss = self.criterion(outputs, batch_label)
                    f_loss.backward()
                    self.optim_f.step()
                
                    train_f_loss.append(f_loss.item())
                    ave_f_loss = np.average(train_f_loss)

                    if (i+1) % 20==0:
                        print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {'%.7f'%ave_f_loss}") 

            test_loss, acc = self.vali()

            print(f"Epoch: {epoch + 1}, cost time: {time.time()-epoch_time} | Test Loss: {'%.7f'%test_loss} Test acc: {'%.4f'%acc}")
            self.scheduler.step()
            early_stopping(acc, {self.model.model: backbone_path, self.model: path})
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        best_model_path = f"{path}/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

    def test(self):
        self.model.load_state_dict(torch.load(f"{self.args.checkpoint_path}/{self.args.setting}/checkpoint.pth"))
        self.model.eval()
        
        preds = []
        trues = []
        embeddings = []
        with torch.no_grad():
            for batch_x, batch_label in self.test_loader:
                batch_x = batch_x.float().to(self.device)
                outputs, embed = self.model(batch_x)
                pred_class = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
                preds.append(pred_class.detach().cpu().numpy())
                trues.append(batch_label.numpy())
                embeddings.append(embed.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        embeddings = np.concatenate(embeddings, axis=0)
        acc = np.sum(trues == preds) / len(trues)
        
        if self.args.plot_tSNE:
            features_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=self.args.tSNE_perplexity).fit_transform(embeddings)
            tsne_result_df = pd.DataFrame({'tsne_1': features_embedded[:,0], 'tsne_2': features_embedded[:,1], 'label': trues})
            fig, ax = plt.subplots(1)
            sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
            lim = (features_embedded.min()-5, features_embedded.max()+5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            mkdir(self.args.fig_save_path)
            fig_path = os.path.join(self.args.fig_save_path, self.args.setting)
            mkdir(fig_path)
            plt.savefig(f"{fig_path}/tSNE.png", dpi=200, bbox_inches='tight')
            plt.close()
        # result save
        folder_path = f'./results/backbone/{self.args.setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'preds.npy', preds)
        np.save(folder_path + 'trues.npy', trues)
        print("test accuracy: {0}".format(acc))


    def visualize_dataset(self):
        _, train_loader = data_provider(self.args, 'train', 'ts', batch_size=1)
        for i, (batch_x, batch_label) in enumerate(train_loader): # [B, D, L]
            x_data = batch_x.flatten()
            fig, ax = plt.subplots(1)
            label = batch_label.flatten().item()
            plt.title(f"class {label}")
            ax.plot(torch.arange(0, x_data.shape[0]), x_data)
            plt.savefig(f"./figs/dataset/class{label}_iter{i}.png", dpi=200, bbox_inches='tight')
            plt.close()


    def model_info(self):  # Plots a line-by-line description of a PyTorch model
        n_p = sum(x.numel() for x in self.model.parameters())  # number parameters
        n_g = sum(x.numel() for x in self.model.parameters() if x.requires_grad)  # number gradients
        print('\n%5s %40s %12s' % ('layer', 'name', 'parameters'))
        param_list = list(self.model.named_parameters())
        for i, (name, p) in enumerate(param_list):
            name = name.replace('module_list.', '')
            print('%5g %40s %12g' % (i, name, p.numel()))
        print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))