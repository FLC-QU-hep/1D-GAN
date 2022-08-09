import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super(Discriminator, self).__init__()

        self.nfd = cfg.nfd

        self.MainConv = nn.Sequential(
            nn.Conv1d(4, cfg.nfd, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd, cfg.nfd*2, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*2),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*2, cfg.nfd*4, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*4),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*4, cfg.nfd*4, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*4),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*4, cfg.nfd*8, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*8),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*8, cfg.nfd*16, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*16),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*16, cfg.nfd*16, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*16),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*16, cfg.nfd*16, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*16),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*16, cfg.nfd*32, 3, 1, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*32),
            nn.LeakyReLU(True),
            
            nn.Conv1d(cfg.nfd*32, cfg.nfd*32, 3, 1, 0, bias=False),
            nn.LeakyReLU(True),
            
        )
        
        self.Classifier = nn.Sequential(
            nn.Linear(cfg.nfd*32,1)
        )

    def forward(self, x):
        x = self.MainConv(x)
        x = x.reshape(-1, self.nfd*32)
        x = self.Classifier(x)
        return x




class DiscriminatorS(nn.Module):
    def __init__(self, cfg):
        super(DiscriminatorS, self).__init__()

        self.nf = cfg.nfd

        self.MainConv = nn.Sequential(
            nn.Conv1d(4, cfg.nfd, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv1d(cfg.nfd, cfg.nfd*2, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*2),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv1d(cfg.nfd*2, cfg.nfd*4, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*4),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv1d(cfg.nfd*4, cfg.nfd*8, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*8),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv1d(cfg.nfd*8, cfg.nfd*16, 3, 3, 0, bias=False),
            nn.BatchNorm1d(cfg.nfd*16),
            nn.LeakyReLU(0.2, True),
            
            nn.Conv1d(cfg.nfd*16, cfg.nfd*32, 3, 3, 0, bias=False),
            nn.LeakyReLU(0.2, True)
            
        )
        
        self.Classifier = nn.Sequential(
            nn.Linear(cfg.nfd*32,1),
            # nn.LeakyReLU(0.2, True),

            # nn.Linear(32,1)
        )

    def forward(self, x):
        x = self.MainConv(x)
        x = x.reshape(-1, self.nf*32)
        x = self.Classifier(x)
        return x