from data import data_provider
from exp.expBasic import Basic
from model.backbones import load_backbone, BaseClassifier
from sklearn.manifold import TSNE
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import accuracy_score
from utils.utils import mkdir
from model.loss import SupConLoss
from model.augment import rand_aug, load_aug
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

class BackboneExp(Basic):
    def __init__(self, args):
        super().__init__(args)
        self.setting = args.setting
        model_list = [model.to(self.device) for model in self._build_model()]
        self.model_g, self.model_f = model_list
    
    def _build_model(self):
        model_g = load_backbone(self.args)
        model_f = BaseClassifier(self.args.d_features, self.args.classifier_d_ffn, self.args.num_classes)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model_g = nn.DataParallel(model_g, device_ids=self.args.device_ids)
            model_f = nn.DataParallel(model_f, device_ids=self.args.device_ids)
        return model_g, model_f

    def _get_data(self, flag, batch_size=None, shuffle=False):
        dataset, dataloader = data_provider(self.args, flag, batch_size, shuffle)
        return dataset, dataloader

    def _select_optimizer(self):
        if self.args.use_contrasive:
            optim_g = optim.SGD(self.model_g.parameters(), lr=0.6)
        else:
            optim_g = optim.Adam(self.model_g.parameters(), lr=self.args.learning_rate)
        optim_f = optim.Adam(self.model_f.parameters(), lr=self.args.learning_rate)
        return optim_g, optim_f
    
    def _select_criterion(self):
        criterion =  nn.CrossEntropyLoss(reduce='mean')
        return criterion

    def vali(self):
        self.model_g.eval()
        self.model_f.eval()
        total_loss = []
        _, test_loader = self._get_data(flag = 'test')
        criterion =  self._select_criterion()
        
        with torch.no_grad():
            for batch_x, batch_label in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.to(self.device)
  
                _, feature = self.model_g(batch_x)
                outputs = self.model_f(feature)
                loss = criterion(outputs, batch_label)
                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model_g.train()
        self.model_f.train()
        return total_loss
    

    def train(self):

        _, train_loader = self._get_data(flag='train', shuffle=True)

        path = os.path.join(self.args.checkpoint_path, self.setting)
        if not os.path.exists(path):
            os.makedirs(path)
        if self.args.resume_train_backbone:
            self.model_g.load_state_dict(torch.load(f"{self.args.checkpoint_path}/{self.setting}/g_checkpoint.pth"))
            self.model_f.load_state_dict(torch.load(f"{self.args.checkpoint_path}/{self.setting}/f_checkpoint.pth"))

        optim_g, optim_f = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_d = SupConLoss(self.args.temperature)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        
        if self.args.use_contrasive:
            for epoch in range(self.args.train_epochs):
                epoch_time = time.time()
                train_c_loss = []
                train_d_loss = []
                self.model_g.train()
                self.model_f.train()
                for i, (batch_x, batch_label) in enumerate(train_loader): # [B, D, L]
                    batch_label = batch_label.to(self.device)
                    aug1, aug2 = rand_aug(batch_x.transpose(-1, -2), batch_label, self.args)
                    aug1 = torch.tensor(aug1).transpose(-1, -2).float().to(self.device)
                    aug2 = torch.tensor(aug2).transpose(-1, -2).float().to(self.device)

                    # update g and c on source
                    optim_g.zero_grad()
                    optim_f.zero_grad()
                    
                    dim = [2*batch_label.shape[0]]
                    dim.extend(list(aug1.shape[1:]))
                    tmp = torch.zeros(tuple(dim))
                    tmp[::2] = aug1.clone()
                    tmp[1::2] = aug2.clone()
        
                    out, feature = self.model_g(tmp.to(self.device))
                    d_loss = criterion_d(out, batch_label)
                    label_tilta = torch.stack((batch_label, batch_label)).T.reshape(-1)
                    outputs = self.model_f(feature)
                    c_loss = criterion(outputs, label_tilta)
                    total_loss = c_loss + self.args.d_loss_eta * d_loss
                    total_loss.backward()
                    optim_f.step()
                    optim_g.step()
                    
                    train_d_loss.append(d_loss.item())
                    ave_d_loss = np.average(train_d_loss)
                    train_c_loss.append(c_loss.item())
                    ave_c_loss = np.average(train_c_loss)
                    if (i+1) % 20==0:
                        print(f"\titers: {i + 1}, epoch: {epoch + 1} | c_loss: {'%.7f'%ave_c_loss} d_loss: {'%.7f'%ave_d_loss}") 

                test_loss = self.vali()

                print(f"Epoch: {epoch + 1}, cost time: {time.time()-epoch_time} | Test Loss: {'%.7f'%test_loss}")
                early_stopping(test_loss, {'f': self.model_f}, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
                adjust_learning_rate(optim_g, 0.6, epoch+1, self.args.lr_min, self.args.contrasive_epochs, decay='cosine')
                adjust_learning_rate(optim_f, self.args.learning_rate, epoch+1, self.args.lr_min, self.args.train_epochs, decay='fixed')
        else:
            for epoch in range(self.args.train_epochs):
                epoch_time = time.time()

                train_c_loss = []
                self.model_g.train()
                self.model_f.train()
                for i, (batch_x, batch_label) in enumerate(train_loader): # [B, D, L]
                    batch_label = batch_label.to(self.device)
                    batch_x = batch_x.to(self.device).float()
                    
                    # update g and c on source
                    optim_g.zero_grad()
                    optim_f.zero_grad()
               
                    out, feature = self.model_g(batch_x)
                    outputs = self.model_f(feature)
                    c_loss = criterion(outputs, batch_label)
                    c_loss.backward()
                    optim_g.step()
                    optim_f.step()
                    
                    train_c_loss.append(c_loss.item())
                    ave_c_loss = np.average(train_c_loss)

                    if (i+1) % 20==0:
                        print(f"\titers: {i + 1}, epoch: {epoch + 1} | c_loss: {'%.7f'%ave_c_loss}") 

                test_loss = self.vali()

                print(f"Epoch: {epoch + 1}, cost time: {time.time()-epoch_time} | Test Loss: {'%.7f'%test_loss}")
                early_stopping(test_loss, {'g': self.model_g, 'f': self.model_f}, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                adjust_learning_rate(optim_g, self.args.learning_rate, epoch+1, self.args.lr_min, self.args.train_epochs, decay='fixed')
                adjust_learning_rate(optim_f, self.args.learning_rate, epoch+1, self.args.lr_min, self.args.train_epochs, decay='fixed')
          
        best_model_g_path = path+'/'+'g_checkpoint.pth'
        best_model_f_path = path+'/'+'f_checkpoint.pth'
        self.model_g.load_state_dict(torch.load(best_model_g_path))
        self.model_f.load_state_dict(torch.load(best_model_f_path))
        

    def test(self):
        _, test_loader = self._get_data(flag='test')
        self.model_g.load_state_dict(torch.load(f"{self.args.checkpoint_path}/{self.setting}/g_checkpoint.pth"))
        self.model_f.load_state_dict(torch.load(f"{self.args.checkpoint_path}/{self.setting}/f_checkpoint.pth"))
    
        self.model_g.eval()
        self.model_f.eval()
        
        preds = []
        trues = []
        embeddings = []
        with torch.no_grad():
            for batch_x, batch_label in test_loader:
                batch_x = batch_x.float().to(self.device)

                _, feature = self.model_g(batch_x)
                outputs = self.model_f(feature)
                pred_class = torch.argmax(F.softmax(outputs), dim=-1)
                
                preds.append(pred_class.detach().cpu().numpy())
                trues.append(batch_label.numpy())
                embeddings.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        embeddings = np.concatenate(embeddings, axis=0)
        acc = accuracy_score(trues, preds)
        
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
            fig_path = os.path.join(self.args.fig_save_path, self.setting)
            mkdir(fig_path)
            plt.savefig(f"{fig_path}/tSNE.png", dpi=200, bbox_inches='tight')
            plt.close()
        # result save
        folder_path = './results/' + self.setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'preds.npy', preds)
        np.save(folder_path + 'trues.npy', trues)
        print("test accuracy: {0}".format(acc))


    def visualize_dataset(self):
        _, train_loader = self._get_data(flag = 'train', batch_size = 1)
        _, test_loader = self._get_data(flag='test')
        for i, (batch_x, batch_label) in enumerate(train_loader): # [B, D, L]
            x_data = batch_x.flatten()
            fig, ax = plt.subplots(1)
            
            aug = load_aug('jitter', batch_x.transpose(-1, -2), batch_label, self.args)
            aug = torch.tensor(aug).transpose(-1, -2).flatten()
            label = batch_label.flatten().item()
            plt.title(f"class {label}")
            ax.plot(torch.arange(0, x_data.shape[0]), x_data)
            ax.plot(torch.arange(0, x_data.shape[0]), aug)
            plt.savefig(f"./figs/dataset/class{label}_iter{i}.png", dpi=200, bbox_inches='tight')
            plt.close()
