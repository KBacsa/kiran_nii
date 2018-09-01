import torch
import torch.nn as nn

class DTAN(nn.Module):
    def __init__(self, n_frames, n_classes):
        super(DTAN, self).__init__()

        self.n_frames = n_frames
        self.n_classes = n_classes
            
        # time convolution
        self.conv1 = nn.Conv2d(in_channels=self.n_frames,
                               out_channels=64*self.n_frames,
                               kernel_size=(5, 5),
                               groups=self.n_frames)
        
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        
        # maxpooling
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        # regular convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(5,5), stride=(1,1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        
        # maxpooling
        self.pool2 = nn.MaxPool2d(2, stride=2)
        
        self.linear1 = nn.Linear(10816, 500)
        self.relu3 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(500, 500)
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(500, self.n_classes)
        
        # softmax
        #self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, x):
           
        N = x.size(0)
        T = x.size(1)
        H = x.size(3)
        W = x.size(4)

        # time convolution
        x = torch.squeeze(x, dim=2)
        x = self.conv1(x)
        x = x.view(N, T, int(x.size(1)/T), x.size(2), x.size(3)).mean(dim=1)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        # regular convolution
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # fully connected
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        #x = self.softmax(x)
        
        return x
    

class DTGN(nn.Module):
    def __init__(self, n_frames, n_classes, n_landmarks):
        super(DTGN, self).__init__()
        
        n_h_1 = int(100 * n_landmarks/49)
        n_h_2 = int(600 * n_landmarks/49)
        
        self.linear1 = nn.Linear(2 * n_landmarks * n_frames, n_h_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.05)
        self.linear2 = nn.Linear(n_h_1, n_h_2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.05)
        self.linear3 = nn.Linear(n_h_2, n_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        #x = self.softmax(x)
        
        return x
    

class Zhang(nn.Module):
    def __init__(self, n_frames, n_classes):
        super(Zhang, self).__init__()
        
        self.time = nn.Conv2d(in_channels=n_frames,
                        out_channels=64*n_frames,
                        kernel_size=(5, 5), padding=(2,2),
                        groups=n_frames)
        
        
        self.features = nn.Sequential(
                            nn.ReLU(),
                            nn.BatchNorm2d(64),
                            nn.MaxPool2d(2, stride=2),
                            nn.Conv2d(64, 96,
                                kernel_size=(5, 5), padding=(2,2)),
                            nn.ReLU(),
                            nn.BatchNorm2d(96),
                            nn.MaxPool2d(2, stride=2),
                            nn.Conv2d(96, 256,
                                kernel_size=(5, 5), padding=(2,2)),
                            nn.ReLU(),
                            nn.MaxPool2d(2, stride=2),
                            nn.Conv2d(256, 256,
                                kernel_size=(5, 5), padding=(2,2)),
                            nn.ReLU()
                        )
        
        
        self.classifier = nn.Sequential(
                              nn.Linear(16384,2048),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(2048,2048),
                              nn.ReLU(),
                              nn.Dropout(0.5),
                              nn.Linear(2048,n_classes)
                            )
    
    
    def forward(self, x):
        
        batch_size = x.size(0)
        T = x.size(1)
        
        x = self.time(x)
        x = x.view(batch_size, T, int(x.size(1)/T), x.size(2), x.size(3)).mean(dim=1)
        x = self.features(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        
        return x