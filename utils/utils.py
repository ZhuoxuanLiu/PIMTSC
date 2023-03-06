import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from utils.dataset import get_UCR_data
from math import ceil, sqrt

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_model_setting(args):
    return f"{args.pr_model}_ds{args.dataset}"

