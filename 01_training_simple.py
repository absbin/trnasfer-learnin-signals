#IMPORT LIBRARIES
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
import sys
from datetime import datetime

from nets import SimpleLSTM

from dataset import SignalDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#SETTINGS
MODEL = SimpleLSTM
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 400
OPTIMIZER = torch.optim.Adam
N_CLASSES = 6

#LOAD DATASETS
metadata_train = pd.read_csv('./data/metadata_train.csv', index_col=0)
metadata_test = pd.read_csv('./data/metadata_test.csv', index_col=0)

dataset_train = SignalDataset(metadata_train, './data/resample_100', mode='train')
dataset_test = SignalDataset(metadata_test, './data/resample_100', mode='test')

loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE//2, shuffle=True)

#INITIALIZE MODEL, CRITERION AND OPTIMIZER
model = MODEL(n_classes=N_CLASSES)
model.initialize_weights() 
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

#TRAIN
model.train()

losses_tr = []
losses_ts = []

for epoch in range(EPOCHS):
    if epoch % 50 == 49:
    	#divide by ten every 50 epochs
        LR/=10
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #for each batch in the dataset
    for j, batch in enumerate(loader_train):
        optimizer.zero_grad()
        
        signal = batch["signal"].to(device)
        target = batch["target"].to(device)
        output = model(signal) 
        loss = criterion(output, target) #compute loss
        loss.backward() #backward
        optimizer.step() #update weights
        loss_tr = loss.item()

        # check loss on valid
        if j % 5 == 0:
            with torch.no_grad():
                batch_ts = next(iter(loader_test))
                signal_ts = batch_ts['signal'].to(device)
                target_ts = batch_ts['target'].to(device)
                output_ts = model.forward(signal_ts)
                loss_ts = criterion(output_ts,target_ts).item()
                
            losses_tr.append(loss_tr)
            losses_ts.append(loss_ts)

        #print status to stdout
        sys.stdout.write('\r Epoch {} of {}  [{:.2f}%] - loss TR/TS: {:.4f}/{:.4f}'.format(epoch+1, EPOCHS, 100*j/len(loader_train), loss_tr, loss_ts))

#SAVE WEIGHTS
now_str = str(datetime.now())[:-7]
torch.save(model.state_dict(), f'./weights/weights_{now_str}.pth')

