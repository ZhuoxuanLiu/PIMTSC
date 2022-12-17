import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import get_UCR_data
from sklearn.preprocessing import StandardScaler


def data_provider(args, flag, type, batch_size=None, shuffle=True):

    if flag == 'train':
        shuffle_flag = shuffle
        drop_last = False
        batch_size = args.batch_size if batch_size is None else batch_size
        if type == 'audio':
            dataset = SCV2_Dataset(args, flag='train')
        else:
            dataset = UCR_Dataset(args, flag='train')
    else:
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        if type == 'audio':
            dataset = SCV2_Dataset(args, flag='test')
        else:
            dataset = UCR_Dataset(args, flag='test')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last)
    return dataset, dataloader


class SCV2_Dataset(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args
        '''
        npy file format:
        {
            "audio_name": str,
            "waveform": (clip_samples,),
            "label": (classes_num,)
        }
        '''
        if flag == 'train':
            self.dataset = np.load(f'datasets/{args.audio_dataset}/train.npy', allow_pickle=True)
        elif flag == 'test':
            self.dataset = np.load(f'datasets/{args.audio_dataset}/test.npy', allow_pickle=True)
        else:
            raise ValueError('flag must be train or test')
        self.total_size = len(self.dataset)
        self.data_x = []
        self.data_y = []
        print(f'Loading {flag} data...')
        for wav_info in tqdm(self.dataset):
            waveform  = torch.tensor(wav_info["waveform"])
            while waveform.shape[0] < self.args.clip_samples:
                waveform = torch.cat((waveform, waveform))
            waveform = waveform[:self.args.clip_samples]
            self.data_x.append(waveform)
            self.data_y.append(wav_info["label"])
        self.data_x = torch.stack(self.data_x, dim=0).unsqueeze_(-1)
        self.data_y = torch.tensor(self.data_y)
          
        
    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return self.total_size
    

class UCR_Dataset(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args
        x_tr, y_tr, x_te, y_te = get_UCR_data(args.dataset, return_split=True, path='.', parent_dir='./datasets/UCR', verbose=False)

        if np.min(y_tr) == 0:
            pass
        
        elif np.min(y_tr) > 0:
            y_tr = y_tr - np.min(y_tr)
            y_te = y_te - np.min(y_te)
        
        elif np.isin(y_tr, [-1, 1]).all():
            y_tr = (y_tr + 1) * 0.5
            y_te = (y_te + 1) * 0.5
        
        else:
            print(y_tr)
            raise Exception("Sorry, transform need to be implemented")
            
            
        assert flag in ['train', 'test']
            
        if flag == 'train':
            self.data_x = x_tr
            self.data_y = y_tr
            del x_tr, y_tr
        else:
            self.data_x = x_te
            self.data_y = y_te
            del x_te, y_te
            
        if args.scale:
            scaler = StandardScaler()
            self.data_x = torch.tensor(self.data_x)
            s, d, l = self.data_x.shape
            x_scale = self.data_x.permute(0, 2, 1).reshape(s*l, d)
            scaler.fit(x_scale)
            self.data_x  = torch.tensor(scaler.transform(x_scale)).reshape(s, l, d).permute(0, 2, 1)
        
        self.data_x = torch.tensor(self.data_x).permute(0, 2, 1)
        self.data_y = torch.tensor(self.data_y).to(torch.int64)
                
        
    def __getitem__(self, index):
        x = self.data_x[index]
        y = self.data_y[index]
        return x, y

    def __len__(self):
        return self.data_x.shape[0]
    
