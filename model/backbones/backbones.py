import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math
import timm
import os
from collections import OrderedDict
from .acousticUtils import MelspectrogramLayer, Normalization2D
import torchaudio
from model.backbones import config as cf
from utils.tools import MappingLayer


class AttRNN(nn.Module):
    
    def __init__(self, args, unet = False):
        super().__init__()
        self.sr = args.sampling_rate
        self.nCategories = 35
        self.unet = unet
        self.acoustic_layer = AcousticLayer(args)
        if self.unet:
            self.up_layer = nn.Sequential(
                nn.Conv2d(1, 16, (5, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(16),
            )
            self.down_layer = nn.Sequential(
                nn.Conv2d(16, 32, (5, 1), padding='same'),
                nn.ReLU(),
                nn.Conv2d(32, 32, (5, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 16, (5, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(16),
            )
            self.merge_layer = nn.Sequential(
                nn.Conv2d(16, 1, (5, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(1),
            )
        else:
            self.net_layer = nn.Sequential(
                nn.Conv2d(1, 10, (5, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(10),
                nn.Conv2d(10, 1, (5, 1), padding='same'),
                nn.ReLU(),
                nn.BatchNorm2d(1),
            )
        self.bi_lstm1 = nn.LSTM(128, 64, 1, batch_first=True, bidirectional=True)
        self.bi_lstm2 = nn.LSTM(128, 64, 1, batch_first=True, bidirectional=True)
        self.qurey_proj = nn.Linear(128, 128)
        self.attn_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, self.nCategories),
        )
        

    def forward(self, x):
        
        x = self.acoustic_layer(x).transpose(1, 3)
        x = Normalization2D(int_axis=0)(x)

        # note that Melspectrogram puts the sequence in shape (batch_size, frames, melDim, input_dim)
        # we would rather have it the other way around for LSTMs

        if self.unet:
            up = self.up_layer(x.permute(0, 3, 1, 2))
            down = self.down_layer(up)
            merge = torch.cat((up, down), dim=-2)
            x = self.merge_layer(merge).permute(0, 2, 3, 1)
        else:
            x = self.net_layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # x = Reshape((125, 80)) (x)
        x = x.squeeze(-1)

        x, _ = self.bi_lstm1(x)  # [b_s, seq_len, vec_dim]
        x, _ = self.bi_lstm2(x)  # [b_s, seq_len, vec_dim]

        xFirst = x[:, -1]  # [b_s, vec_dim]
        query = self.qurey_proj(xFirst)

        # dot product attention
        attScores = torch.einsum('bv, bsv -> bs', query, x)
        attScores = F.softmax(attScores, dim=1)  # [b_s, seq_len]

        # rescale sequence
        attVector = torch.einsum('bs, bsv -> bv', attScores, x) # [b_s, vec_dim]

        out = self.attn_proj(attVector)

        return out
    
# Adverserial Reprogramming layer
class ARTLayer(nn.Module):
    def __init__(self, dropout=0.4):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.build = False

    def _bulid(self, x):
        _, input_len, input_dim = x.shape
        W = torch.zeros(input_len, input_dim).to(x)
        self.W = nn.init.xavier_uniform_(W)
        self.build = True

    def forward(self, x):
        if not self.build:
            self._bulid(x)
        prog = self.dropout(self.W)
        out = x + prog
        return out


class FullAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(FullAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        E = keys.shape[2]
        scale = 1. / torch.sqrt(E)

        scores = torch.einsum("ble,bse->bls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bls,bsd->bld", A, values)

        return V
    
    
class AttentionLayer(nn.Module):
    def __init__(self, d_model, d_keys=None, d_values=None, dropout=0.1):
        super(AttentionLayer, self).__init__()
        self.inner_attention = FullAttention(dropout)
        self.query_projection = nn.Linear(d_model, d_keys)
        self.key_projection = nn.Linear(d_model, d_keys)
        self.value_projection = nn.Linear(d_model, d_values)
        self.out_projection = nn.Linear(d_values, d_model)

    def forward(self, queries, keys, values):
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)

        out = self.inner_attention(queries, keys, values)

        return self.out_projection(out)
    
class SegZeroPadding(nn.Module):
    def __init__(self, seg_num):
        super().__init__()
        self.seg_num = seg_num
        
    def forward(self, x):
        batch_size, input_len, input_dim = x.shape
        src_xlen = 16000
        all_seg = src_xlen//input_len
        seg_len = np.int(np.floor(all_seg//self.seg_num))
        aug_x = torch.zeros(batch_size, src_xlen, input_dim).to(x.device)
        for s in range(self.seg_num):
            startidx = (s*seg_len)*input_len
            endidx = (s*seg_len)*input_len + input_len
            # print('seg idx: {} --> start: {}, end: {}'.format(s, startidx, endidx))
            seg_x = F.pad(x, (0, 0, startidx, src_xlen-endidx))
            aug_x += seg_x
        return aug_x
    

class ProjectionLayer(nn.Module):
    def __init__(self, proj_method, d_features, n_mels, pr_model):
        super().__init__()
        self.proj_method = proj_method
        self.d_features = d_features
        F = math.ceil(n_mels//32)
        if self.proj_method == 'linear':  
            # avg_pool-> [batch_size, channels] -> [batch_size, d_features]
            self.proj = nn.Linear(cf.output_channel[pr_model], self.d_features)
        elif self.proj_method == 'conv':
            # proj-> [batch_size, d_features, 1, ceil(frames//32)] -> [batch_size, d_features]
            self.proj = nn.Conv2d(
                in_channels = cf.output_channel[pr_model],
                out_channels = self.d_features,
                kernel_size = (F,3),
                padding = (1,0)
                )
        elif self.proj_method == 'attn':
            # avg_pool-> [batch_size, channels, ceil(frames//32)] -> [batch_size, d_features]
            self.proj = AttentionLayer(cf.output_channel[pr_model], cf.output_channel[pr_model], self.d_features)
    
        
    def forward(self, x):
        B, C, T, F = x.shape
        if self.proj_method == 'linear':
            out = torch.flatten(x, 2)
            out = torch.mean(out, dim=-1)
            out = self.proj(out)
        elif self.proj_method == 'conv':
            out = torch.flatten(self.proj(x), 2)
            out = torch.mean(out, dim=-1)
        elif self.proj_method == 'attn':
            out = x.reshape(B, -1, F)
            out = torch.mean(out, dim=-1)
            out = out.reshape(B, C, T).permute(0, 2, 1)
            out = self.proj(out, out, out) # [B, T, D]
            out = torch.mean(out, dim=1)
        return out
    
    
class MyModel(nn.Module):
    def __init__(self, args, load=True):
        super().__init__()
        self.args = args
        self.padding_layer = SegZeroPadding(args.seg_num)
        # self.padding_layer = InterpolatePadding()
        self.ART_layer = ARTLayer(args.dropout)
        if args.pr_model != 'attRNN':
            self.pr_model = ImageModel(args, load=False)
            self._pr_adjust()
        else:
            self.pr_model = AttRNN(args)
        if load:
            self._load()
        
    def _pr_adjust(self):
        self.pr_model = nn.Sequential(OrderedDict([
            ('acoustic_layer', list(self.pr_model.children())[0]),
            ('model', list(self.pr_model.children())[1]),
            ('proj_layer', list(self.pr_model.children())[2]),
            # ('classifier', nn.Linear(self.args.d_features, 35))
            ]))
        # for p in self.pr_model.parameters():
        #     p.requires_grad = False
        
    def _load(self):
        load_path = f"{self.args.pretrained_path}/{self.args.pr_model}_ds{self.args.audio_dataset}/checkpoint.pth"
        if os.path.exists(load_path):
            pr_model_dict = torch.load(load_path)
            if 'classifier.weight' in pr_model_dict.keys() and 'classifier.bias' in pr_model_dict.keys():
                pr_model_dict.pop('classifier.weight')
                pr_model_dict.pop('classifier.bias')
            input_convs = self.pr_model.model.pretrained_cfg.get('first_conv', None)
            if isinstance(input_convs, str):
                input_convs = (input_convs,)
            for input_conv_name in input_convs:
                weight_name = f"model.{input_conv_name}.weight"
                conv_weight = pr_model_dict[weight_name]
                O, I, J, K = conv_weight.shape
                if self.args.c_in != 1:
                    repeat_num = int(self.args.c_in)
                    conv_weight = conv_weight.repeat(1, repeat_num, 1, 1)
                    conv_weight *= (1 / float(self.args.c_in))
                pr_model_dict[weight_name] = conv_weight
            self.pr_model.load_state_dict(pr_model_dict, strict=True)
                            
    def forward(self, x):  
        x_aug = self.padding_layer(x)

        x_aug = self.ART_layer(x_aug) # e.g., input_shape[0] = 500 for FordA
        
        out = self.pr_model(x_aug)   

        return out
    

class MyModelWarpper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = MyModel(args, load=True)
        if args.pr_model != 'attRNN':
            self.classifier = nn.Linear(args.d_features, args.num_classes)
        else:
            self.classifier = nn.Linear(35, args.num_classes)
        # self.classifier = RNNLayer(args.d_features, args.num_classes)
        
    def forward(self, x):
        embedding = self.model(x)
        out = self.classifier(embedding)
        return out, embedding
    
    
class AcousticLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.m = MelspectrogramLayer(n_fft=1024, hop_length=160, center=True, pad_begin=False, 
                                     sample_rate=args.sampling_rate, n_mels=args.n_mels, pow=1.0, mel_f_min=40.0, 
                                     mel_f_max=args.sampling_rate / 2, return_decibel=True)
        
    def forward(self, x):
        # x: (batch, time, dim)
        x = self.m(x)  # (batch, frame, nmels, dim)
        x = x.permute(0, 3, 2, 1)  # (batch, dim, nmels, frame)
        if (self.args.freqm > 0 or self.args.timem > 0) and self.training:
            if self.args.freqm > 0:
                freq_mask = torchaudio.transforms.FrequencyMasking(self.args.freqm)
                x = freq_mask(x)
            if self.args.timem > 0:
                time_mask = torchaudio.transforms.TimeMasking(self.args.timem)
                x = time_mask(x)
        return x
    
    
class ImageModel(nn.Module):
    
    def __init__(self, args, load=True) -> None:
        super().__init__()
        self.acoustic_layer = AcousticLayer(args)
        # self.norm_layer = Normalization2D(int_axis=0)
        self.model = eval(cf.create_model[args.pr_model])
        self.proj_layer = ProjectionLayer(args.proj_method, args.d_features, args.n_mels, args.pr_model)
        self.model.global_pool = nn.Identity()
        self.model.classifier = nn.Identity()
        self.args = args
        self.build = False
        if load:
            self._load()
        
    def _load(self):
        pr_model_dict = torch.load(cf.checkpoint[self.args.pr_model])
        if 'classifier.weight' in pr_model_dict.keys() and 'classifier.bias' in pr_model_dict.keys():
            pr_model_dict.pop('classifier.weight')
            pr_model_dict.pop('classifier.bias')
        input_convs = self.model.pretrained_cfg.get('first_conv', None)
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = f"{input_conv_name}.weight"
            conv_weight = pr_model_dict[weight_name]
            O, I, J, K = conv_weight.shape
            if self.args.c_in == 1:
                if I > 3:
                    assert conv_weight.shape[1] % 3 == 0
                    # For models with space2depth stems
                    conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
                    conv_weight = conv_weight.sum(dim=2, keepdim=False)
                else:
                    conv_weight = conv_weight.sum(dim=1, keepdim=True)
            elif self.args.c_in != 3:
                repeat_num = int(math.ceil(self.args.c_in / 3))
                conv_weight = conv_weight.repeat(1, repeat_num, 1, 1)[:, :in_chans, :, :]
                conv_weight *= (3 / float(self.args.c_in))
            pr_model_dict[weight_name] = conv_weight
        self.model.load_state_dict(pr_model_dict)
        
    def forward(self, x):
        x = self.acoustic_layer(x)
        # x: (batch, dim, nmels, frames)
        # x = self.norm_layer(x)
        x = self.model(x)
        out = self.proj_layer(x)
        return out
    

class ImageModelWarpper(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = ImageModel(args, load=True)
        self.classifier = nn.Linear(args.d_features, args.num_classes)
        
    def forward(self, x):
        embedding = self.model(x)
        out = self.classifier(embedding)
        return out, embedding