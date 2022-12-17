from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
import torch
import math

def lr_adjust(epoch):
    lr = math.cos((epoch/(20*eta))*(math.pi/2)) * 1
    print(lr)
    
if __name__ == "__main__":

    # # result save
    # folder_path = './results/resnetv2_50_bit_fd512_dsMiddlePhalanxTW/'

    # preds = np.load(folder_path + 'preds.npy')
    # trues = np.load(folder_path + 'trues.npy')
    # acc = accuracy_score(trues, preds)
    # print("test accuracy: {0}".format(acc))
    
    lr_adjust(10)

