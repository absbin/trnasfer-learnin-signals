import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, n_classes=2):
        super(SimpleLSTM, self).__init__()
        self.n_classes = n_classes
        
        self.conv_branch = nn.Sequential(
            nn.Conv1d(1, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, 3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, 3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(10)
        )

        self.lstm = nn.LSTM(256, 100, 1)

        self.linear = nn.Sequential(nn.Linear(100, self.n_classes),
                                    nn.Softmax(1))
        
    def forward(self, x):
        x_feat = self.conv_branch(x)
        x_feat = x_feat.permute(2,0,1)
        x_lstm, _ = self.lstm(x_feat)
        x_out = self.linear(x_lstm[-1])
        return(x_out)

    def extract_features(self, x):
        x_feat = self.conv_branch(x)
        x_feat = x_feat.permute(2,0,1)
        x_lstm, _ = self.lstm(x_feat)
        return(x_lstm[-1])
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)