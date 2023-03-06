import argparse
import os
import torch
from exp import ModelExp, BackboneExp
from utils.utils import get_model_setting
from utils.dataset import get_dataset_desc
import random
import numpy as np
import matplotlib


def main():
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description='God please help me')

    # basic config
    parser.add_argument('--train_backbone', action='store_true', default=False, help='train backbone')
    parser.add_argument('--resume', action='store_true', default=False, help='train backbone')
    parser.add_argument('--test_backbone', action='store_true', default=False, help='test backbone')
    parser.add_argument('--train_model', action='store_true', default=False, help='train model')
    parser.add_argument('--test_model', action='store_true', default=False, help='test model')
    parser.add_argument('--plot_tSNE', action='store_true', default=False, help='plot tSNE')
    parser.add_argument('--tSNE_perplexity', type=int, default=30, help='tSNE perplexity')
    parser.add_argument('--visualize_data', action='store_true', default=False, help='tSNE perplexity')
    parser.add_argument('--use_metric', action='store_true', default=False, help='use metric learning')
    
    # backbone
    parser.add_argument('--pr_model', type=str, default='efficientnetv2_rw_t')
    parser.add_argument('--proj_method', type=str, default='linear', help='projection method')
    
    # data loader
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--scale', action='store_true', default=False, help='normalize data')
    parser.add_argument('--batch_size', type=int, default=48, help='batch size of train input data')

    # model define
    
    # general
    parser.add_argument('--d_features', type=int, default=1024, help='dimension of embed features')
    parser.add_argument('--proj_features', type=int, default=128, help='dimension of projected features')
    
    # audio
    parser.add_argument('--seg_num', type=int, default=18)
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
    parser.add_argument('--backbone_path', type=str, default='./checkpoints/backbones', help='location of backbone')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/models', help='location of model checkpoints')
    parser.add_argument('--model_save_path', type=str, default='./save', help='location of model save')
    parser.add_argument('--fig_save_path', type=str, default='./figs', help='location of fig save')
    
    # optimization
    parser.add_argument('--dropout', type=float, default=0.4, help='dropout')
    parser.add_argument('--temperature', type=float, default=0.07, help='supervised contrast loss temp')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='lr')
    parser.add_argument('--lr_min', type=float, default=1e-4, help='lr')
    parser.add_argument('--eta', type=float, default=0.1, help='eta')
    
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

    args.c_in, args.seq_len, args.num_classes, avg_class_examples = get_dataset_desc(args.dataset)

    args.setting = get_model_setting(args)
    
    if args.train_backbone or args.test_backbone:
        backbone_exp = BackboneExp(args)  # set experiments
        if args.visualize_data:
            backbone_exp.visualize_dataset()
    if args.train_backbone:
        # setting record of experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.setting))
        backbone_exp.train()

        torch.cuda.empty_cache()
        
            
    if args.test_backbone:
        print('>>>>>>>backbone testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))
        backbone_exp.test()
    
    if args.train_model or args.test_model:
        model_exp = ModelExp(args)
    if args.train_model:
        model_exp.train()

    # setting record of experiments
    if args.test_model:
        print('>>>>>>>model testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.setting))
        model_exp.test()
        
if __name__ == "__main__":
    main()
