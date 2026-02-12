import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def initialBlocks(self, dropoutP):
        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.C1),
                      padding=(0, self.C1 // 2), bias=False),
            nn.BatchNorm2d(self.F1),
            
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=self.F1),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutP))

        block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 22),
                      padding=(0, 22 // 2), bias=False,
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                      stride=1, bias=False, padding=0),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutP))
        
        return nn.Sequential(block1, block2)

    def calculateOutSize(self, model, nChan, nTime):
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        with torch.no_grad():
            out = model(data).shape
        return out[2:] 

    def __init__(self, nChan, nTime, embedDim=768, 
                 dropoutP=0.25, F1=8, D=2, C1=64):
        super(EEGNet, self).__init__()
        self.F2 = D * F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nChan = nChan
        self.C1 = C1
        self.embedDim = embedDim

        self.firstBlocks = self.initialBlocks(dropoutP)
        
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)

        self.projection = nn.Sequential(
            nn.Conv2d(self.F2, embedDim, (1, self.fSize[1]), bias=True),
            nn.Flatten()
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.firstBlocks(x)
        x = self.projection(x)
        return x
