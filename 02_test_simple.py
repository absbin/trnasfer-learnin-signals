#IMPORT LIBRARIES
import torch
import numpy as np
import pandas as pd
from nets import SimpleLSTM
from dataset import SignalDataset
from sklearn.metrics import matthews_corrcoef as mcc, accuracy_score as acc, confusion_matrix

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

#PREDICTIONS ON TRAIN
model.eval()

preds = []
trues = []
probs = []
filenames = []
with torch.no_grad():
    for sample in dataset_train:
        signal = sample["signal"].unsqueeze(0).to(device)
        target = sample["target"]
        output = model(signal) #forward
        _, pred = torch.max(output,1)
        
        preds.append(pred.data.cpu().numpy())
        trues.append(target)
        probs.append(output.data.cpu().numpy())
        filenames.append(sample['file'])

probs_tr = np.concatenate(probs)
preds_tr = np.concatenate(preds)
trues_tr = np.array(trues)
filenames_tr = np.array(filenames)

#GET PERFORMANCES ON TRAIN
MCC = mcc(trues_tr, preds_tr)
ACC = acc(trues_tr, preds_tr)
print("MCC train", MCC, "ACC train", ACC)

print(confusion_matrix(trues_tr, preds_tr))

#PREDICTIONS ON TEST
preds = []
trues = []
probs = []
filenames = []
with torch.no_grad():
    for sample in dataset_test:
        signal = sample["signal"].unsqueeze(0).to(device)
        target = sample["target"]
        output = model(signal) #forward
        _, pred = torch.max(output,1)
        
        preds.append(pred.data.cpu().numpy())
        trues.append(target)
        probs.append(output.data.cpu().numpy())
        filenames.append(sample['file'])

probs_ts = np.concatenate(probs)
preds_ts = np.concatenate(preds)
trues_ts = np.array(trues)
filenames_ts = np.array(filenames)

#GET PERFORMANCES ON TEST
MCC = mcc(trues_ts, preds_ts)
ACC = acc(trues_ts, preds_ts)
print("MCC test", MCC, "ACC test", ACC)

print(confusion_matrix(trues_ts, preds_ts))