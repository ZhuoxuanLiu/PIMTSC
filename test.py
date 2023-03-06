import torch
import timm
from model.backbones import MyModelWarpper
import argparse
from model.backbones import ImageModel
from collections import OrderedDict

parser = argparse.ArgumentParser(description='God please help me')

# basic config
parser.add_argument('--train_backbone', action='store_true', default=False, help='train backbone')
parser.add_argument('--resume_train_backbone', action='store_true', default=False, help='train backbone')
parser.add_argument('--test_backbone', action='store_true', default=False, help='test backbone')
parser.add_argument('--train_model', action='store_true', default=False, help='train model')
parser.add_argument('--test_model', action='store_true', default=False, help='test model')
parser.add_argument('--plot_tSNE', action='store_true', default=False, help='plot tSNE')
parser.add_argument('--tSNE_perplexity', type=int, default=30, help='tSNE perplexity')
parser.add_argument('--visualize_data', action='store_true', default=False, help='tSNE perplexity')
parser.add_argument('--use_contrasive', action='store_true', default=False, help='use contrasive')

# backbone
parser.add_argument('--backbone', type=str, default='efficientnetv2_rw_t')
parser.add_argument('--proj_method', type=str, default='linear', help='projection method')

# data loader
parser.add_argument('--dataset', type=str, default='UCR', help='dataset name')
parser.add_argument('--scale', action='store_true', default=False, help='normalize data')
parser.add_argument('--batch_size', type=int, default=48, help='batch size of train input data')

# model define

# general
parser.add_argument('--pr_model', type=str, default='efficientnetv2_rw_t')
parser.add_argument('--d_features', type=int, default=1024, help='dimension of embed features')
parser.add_argument('--proj_features', type=int, default=128, help='dimension of projected features')

# audio
parser.add_argument('--seg_num', type=int, default=3)
parser.add_argument('--clip_samples', type=int, default=16000)
parser.add_argument('--sampling_rate', type=int, default=16000)
parser.add_argument('--n_mels', type=int, default=128)
parser.add_argument('--audio_dataset', type=str, default='scv2')


# nearest neighbour search
parser.add_argument('--nn_method', type=str, default='faiss', help='nearest neighbour method')
parser.add_argument('--n_neighbours', type=int, default=8, help='number of neighbours')
parser.add_argument('--faiss_workers', type=int, default=4, help='number of faiss workers')
parser.add_argument('--update_in_test', action='store_true', default=False, help='update feature bank in test')
parser.add_argument('--update_bank_interv', type=int, default=10, help='update feature bank interval')

# sampler
parser.add_argument('--sampler', type=str, default='greedy_coreset', help='sampler')
parser.add_argument('--sample_percentage', type=float, default=0.8, help='sample percentage')
parser.add_argument('--n_starting_points', type=int, default=4, help='approximate coreset starting points')

# augmentation
parser.add_argument('--freqm', type=int, default=24)
parser.add_argument('--timem', type=int, default=24)
# ts aug
parser.add_argument('--aug_jitter_sigma', type=float, default=0.03, help='jitter sigma')
parser.add_argument('--aug_scale_sigma', type=float, default=0.1, help='scale sigma')
parser.add_argument('--aug_magwarp_sigma', type=float, default=0.2, help='magnitude warp sigma')
parser.add_argument('--aug_magwarp_knot', type=int, default=4, help='magnitude warp knot')
parser.add_argument('--aug_permu_segments', type=int, default=5, help='permutation segments')
parser.add_argument('--aug_permu_seg_mode', type=str, default='equal', help='permutation mode: equal or random')
parser.add_argument('--aug_time_warp_sigma', type=float, default=0.2, help='time warp sigma')
parser.add_argument('--aug_time_warp_knot', type=int, default=4, help='time warp knot')
parser.add_argument('--aug_window_slice_ratio', type=float, default=0.9, help='window reduce_ratio')
parser.add_argument('--aug_window_warp_ratio', type=float, default=0.1, help='window warp ratio')
parser.add_argument('--aug_window_warp_scale', type=str, default='[0.5,2]', help='window warp scale range')
parser.add_argument('--aug_batch_size', type=int, default=3, help='augmentation batch size')
parser.add_argument('--aug_slope_constraint', type=str, default='symmetric', help='slope_constraint is for DTW: symmetric or asymmetric')
parser.add_argument('--aug_use_window', type=int, default=1, help='use window')

# save and load
parser.add_argument('--pretrained_path', type=str, default='./pretrained', help='location of pretrained checkpoints')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='location of model checkpoints')
parser.add_argument('--model_save_path', type=str, default='./save', help='location of model save')
parser.add_argument('--fig_save_path', type=str, default='./figs', help='location of fig save')

# optimization
parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
parser.add_argument('--temperature', type=float, default=0.07, help='supervised contrast loss temp')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.01, help='lr')
parser.add_argument('--lr_min', type=float, default=1e-4, help='lr')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.c_in, args.seq_len, args.num_classes, avg_class_examples = 1, 100, 2, 100
    
# imodel = ImageModel(args, load=False)

# im_dict = torch.load('pretrained/gc_efficientnetv2_rw_t_dsscv2/checkpoint.pth')
# new_dict = []
# for k in im_dict.keys():
#     if 'model.model' in k:
#         new_dict.append((k.partition('.')[-1], im_dict[k]))
#     elif 'model.proj_layer' in k:
#         new_dict.append((k.partition('.')[-1], im_dict[k]))
#     else:
#         new_dict.append((k, im_dict[k]))
# new_dict = OrderedDict(new_dict)
# torch.save(new_dict, 'pretrained/gc_efficientnetv2_rw_t_dsscv2/checkpoint.pth')

# model = MyModelWarpper(args)
# print(model)

# model_dict = torch.load("pretrained/efficientnetv2_rw_t_dsscv2/checkpoint.pth")
# print(list(model_dict.keys())[-5:])
# print(model_dict['proj.weight'].shape)


import numpy as np
from sktime.classification.hybrid import HIVECOTEV2
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.deep_learning import CNNClassifier
from sktime.classification.kernel_based import RocketClassifier
from sktime.transformations.panel.rocket import MiniRocket
from sktime.datasets import load_arrow_head
from utils.dataset import get_UCR_data
from sklearn.metrics import accuracy_score

X_train, y_train, X_test, y_test = get_UCR_data('GunPoint', return_split=True, path='.', parent_dir='./datasets/UCR', verbose=False)

if np.min(y_train) == 0:
    pass
        
elif np.min(y_train) > 0:
    y_train = y_train - np.min(y_train)
    y_test = y_test - np.min(y_test)

elif np.isin(y_train, [-1, 1]).all():
    y_train = (y_train + 1) * 0.5
    y_test = (y_test + 1) * 0.5
            
pre_clf = MiniRocket()
x_transformed = pre_clf.fit_transform(X_train, y=None)
print(x_transformed.shape)
# y_pred = classifier.predict(X_test)
# print(accuracy_score(y_test, y_pred))
