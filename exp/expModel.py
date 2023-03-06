from data import data_provider
from exp.expBasic import Basic
from model.model import Model

from utils.metrics import accuracy_score
import numpy as np

import torch
import torch.nn as nn

import os
import time

import warnings
warnings.filterwarnings('ignore')

class ModelExp(Basic):
    def __init__(self, args):
        super().__init__(args)
        self.setting = args.setting
        self.model = self._build_model()
        if args.train_model:
            self.model.load(args, self.device, self.setting)
        else:
            self.model.load_from_path(args.model_save_path, self.device, self.setting)
        self.model = self.model.to(self.device)
    
    def _build_model(self):
        model = Model()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        dataset, dataloader = data_provider(self.args, flag, 'ts')
        return dataset, dataloader


    def train(self):
        _, train_loader = self._get_data(flag = 'train')
        self.model.fit(train_loader)
        self.model.save_to_path(self.args.model_save_path)

    def test(self):
        _, test_loader = self._get_data(flag='test')
        
        pred, gt = self.model.predict(test_loader)

        acc = accuracy_score(pred.numpy(), gt.numpy())
        # result save
        folder_path = f'./results/model/{self.setting}/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'preds.npy', pred)
        np.save(folder_path + 'trues.npy', gt)
        print("test accuracy: {0}".format(acc))
