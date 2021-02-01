import numpy as np
import torch
from torch.utils.data import Dataset
import os

signal_dict = {'ECG': 0,
               'EDA': 1,
               'EMG': 2,
               'PPG': 3,
               'RESP': 4,
               'ACC' : 5}

device_dict = {'Biosemi': 0,
               'Bitalino': 1,
               'E4': 2,
               'FlexComp': 3}

def normalize(x):
    if np.std(x) != 0:
        return( (x - np.mean(x)) / np.std(x) )
    else:
        return(x - np.mean(x))

#%%
class SignalDataset(Dataset):
    def __init__(self, metadata, rootdir, N = 1000, mode = 'train', seed = 123):
        np.random.seed(seed)
        self.metadata = metadata
        self.rootdir = rootdir
        self.mode = mode
        self.N = N
        
    def __len__(self):
        return(self.metadata.shape[0])
    
    def __getitem__(self, idx):
        signal_row = self.metadata.iloc[idx, :]
        
        subset_dir = signal_row['Dataset']
        signal_dir = signal_row['SignalName']
        device = signal_row['Device']
        filename = signal_row.name.split('.')[0]+'.npz'
        
        signal_file = os.path.join(self.rootdir, subset_dir, signal_dir, filename)
        
        #load signal
        signal = np.load(signal_file)['signal']
        
        if self.mode == 'train':
            #randomly select portion (=start_idx)
            max_idx = len(signal) - self.N
            start_idx = np.random.randint(0, max_idx)
        else:
            #take the central portion (=start_idx)
            start_idx = int((len(signal) - self.N)/2)
            
        signal_portion = signal[start_idx: start_idx+self.N].reshape(1, self.N)
        signal_portion = normalize(signal_portion)        
        signal_portion = torch.tensor(signal_portion).float()
        
        return({'signal':signal_portion, 'target': signal_dict[signal_dir], 
                'signal_type': signal_dir, 'file':signal_file,
                'device_target':device_dict[device], 'device_type': device,
        })