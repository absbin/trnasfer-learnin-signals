#IMPORT LIBRARIES
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from nets import SimpleLSTM
from dataset import SignalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#SETTINGS
MODEL = SimpleLSTM
N_CLASSES = 6
weights_file = 'weights_2020-12-30 11:02:34.pth'

#LOAD DATASETS
metadata_train = pd.read_csv('./data/metadata_train.csv', index_col=0)
metadata_test = pd.read_csv('./data/metadata_test.csv', index_col=0)

dataset_train = SignalDataset(metadata_train, './data/resample_100', mode='test')
dataset_test = SignalDataset(metadata_test, './data/resample_100', mode='test')

#INITIALIZE MODEL WITH TRAINED WEIGHTS
state_dict = torch.load(f'./weights/{weights_file}', map_location=device)

model = MODEL(n_classes=N_CLASSES)
model.load_state_dict(state_dict)
model = model.to(device)

#EXTRACT FEATURES ON TRAIN
model.eval()

features_train = []
type_train = []
device_train = []
filenames_train = []
with torch.no_grad():
    for sample in dataset_train:
        signal = sample["signal"].unsqueeze(0).to(device)
        signal_type = sample["signal_type"]
        device_type = sample["device_type"]
        features = model.extract_features(signal) #forward
        features_train.append(features.data.cpu().numpy())
        type_train.append(signal_type)
        device_train.append(device_type)
        filenames_train.append(sample['file'])
        
features_train = np.array(features_train)[:,0,:]
type_train = np.array(type_train)
device_train = np.array(device_train)

#EXTRACT FEATURES ON TEST
features_test = []
type_test = []
device_test = []
filenames_test = []
with torch.no_grad():
    for sample in dataset_test:
        signal = sample["signal"].unsqueeze(0).to(device)
        signal_type = sample["signal_type"]
        device_type = sample["device_type"]
        features = model.extract_features(signal) #forward
        features_test.append(features.data.cpu().numpy())
        type_test.append(signal_type)
        device_test.append(device_type)
        filenames_test.append(sample['file'])

features_test = np.array(features_test)[:,0,:]
type_test = np.array(type_test)
device_test = np.array(device_test)

features = np.concatenate([features_train, features_test])
signal_type = np.concatenate([type_train, type_test])
device_type = np.concatenate([device_train, device_test])

#COMPUTE PCA
pca = PCA(n_components=2)

pca_feat = pca.fit_transform(features)

#PLOT PCA RESULTS
f, axes = plt.subplots(1,2)
f.set_size_inches(10,5)
plt.sca(axes[0])
for s_type in np.unique(signal_type):
  idx_type = np.where(signal_type == s_type)[0]
  plt.plot(pca_feat[idx_type, 0], pca_feat[idx_type, 1], '.')#, alpha=0.5)

plt.grid()
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 1')

plt.legend(np.unique(signal_type))

plt.sca(axes[1])
for d_ in np.unique(device_type):
  idx_device = np.where(device_type == d_)[0]
  plt.plot(pca_feat[idx_device, 0], pca_feat[idx_device, 1], '.')#, alpha=0.5)
plt.grid()
plt.xlabel('PCA component 1')
plt.ylabel('PCA component 1')

plt.legend(np.unique(device_type))

