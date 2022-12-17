import torch
import torch.nn as nn
import numpy as np
import math

    
class MappingLayer(nn.Module):
    
    def __init__(self, source_num, mapping_num, target_num):
        super().__init__()
        self.source_num = source_num
        self.mapping_num = mapping_num
        self.target_num = target_num
        mt = self.mapping_num * self.target_num ##mt must smaller than source_num
        self.label_map = torch.zeros(self.source_num, mt) ##[source_num, map_num*target_num]
        self.label_map[0:mt, 0:mt] = torch.eye(mt) ##[source_num, map_num*target_num]
        
    def forward(self, prob):
        map_prob = torch.matmul(prob, self.label_map.to(prob)) ## [1, source_num] * [source_num, map_num*target_num] = [1, map_num*target_num]
        final_prob = torch.mean(map_prob.reshape(map_prob.shape[0], self.target_num, self.mapping_num), dim=-1) ##[target_num]
        return final_prob 


def adjust_learning_rate(optimizer, epoch, args, type='adam'):
    if type == 'sgd':
        lr_adjust = {epoch: args.lr_min+0.5*(args.learning_rate-args.lr_min)*(1+math.cos((epoch/args.train_epochs)*math.pi))}
    elif type == 'adam':
        lr_adjust = {
            2: 5e-3, 6: 1e-3, 10: 5e-4, 15: 1e-4, 20: 5e-5
        }
    if epoch in lr_adjust.keys():
        lr_adjusted = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_adjusted
        print('Updating learning rate to {}'.format(lr_adjusted))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.acc_min = np.Inf
        self.delta = delta

    def __call__(self, acc, model, path):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(acc, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(acc, model, path)
            self.counter = 0

    def save_checkpoint(self, acc, model: dict, path):
        if self.verbose:
            print(f'accuracy increased ({self.acc_min:.6f} --> {acc:.6f}).  Saving model ...')
        for k, v in model.items():
            torch.save(v.state_dict(), f"{path}/checkpoint.pth")
        self.acc_min = acc
        

class OptimWrapper:
    "Optim wrapper that implements rate."
    def __init__(self, lr, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.lr = lr
        self._rate = 0
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (self.lr ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))) 