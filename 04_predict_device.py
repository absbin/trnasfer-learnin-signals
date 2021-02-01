#IMPORT LIBRARIES
import torch
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from nets import SimpleLSTM
from dataset import SignalDataset
from sklearn.metrics import matthews_corrcoef as mcc, confusion_matrix

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

#EXTRACT FEATURES FROM TRAIN
model.eval()

features_train = []
labels_train = []
filenames_train = []
with torch.no_grad():
    for sample in dataset_train:
        signal = sample["signal"].unsqueeze(0).to(device)
        target = sample["device_target"]
        features = model.extract_features(signal) #forward
        features_train.append(features.data.cpu().numpy())
        labels_train.append(target)
        filenames_train.append(sample['file'])
        
features_train = np.array(features_train)[:,0,:]
labels_train = np.array(labels_train)

#EXTRACT FEATURES FROM TEST
features_test = []
labels_test = []
filenames_test = []
with torch.no_grad():
    for sample in dataset_test:
        signal = sample["signal"].unsqueeze(0).to(device)
        target = sample["device_target"]
        features = model.extract_features(signal) #forward
        features_test.append(features.data.cpu().numpy())
        labels_test.append(target)
        filenames_test.append(sample['file'])

features_test = np.array(features_test)[:,0,:]
labels_test = np.array(labels_test)

#PREDICT DEVICE WITH A SIMPLE SVM
svm = SVC()

svm.fit(features_train, labels_train)

pred_train = svm.predict(features_train)
pred_test = svm.predict(features_test)

#GET PERFORMANCES
print(mcc(labels_train, pred_train))
print(mcc(labels_test, pred_test))

print(confusion_matrix(labels_train, pred_train))
print(confusion_matrix(labels_test, pred_test))

