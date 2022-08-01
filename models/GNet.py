import torch
import torch.nn as nn



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)



class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.TransConv = nn.Sequential(

            nn.ConvTranspose1d(1024, 8192, 2, 1, 0, bias=False),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(8192, 4096, 3, 3, 0, bias=False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(4096, 2048, 3, 3, 1, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(2048, 1024, 3, 3, 0, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(True)
        )
        
        self.CoordinatesPredictor = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.ConvTranspose1d(256, 128, 3, 3, 0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 64, 3, 3, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.ConvTranspose1d(64, 32, 3, 3, 0, bias=False),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),

            nn.ConvTranspose1d(32, 16, 3, 3, 0, bias=False),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),

            nn.ConvTranspose1d(16, 2, 9, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        
        self.EnergyPredictor = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 3, 3, 0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 64, 3, 3, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, 3, 3, 0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 16, 3, 3, 0, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),

            nn.ConvTranspose1d(16, 1, 9, 1, 0, bias=False),
            nn.ReLU()
        )
        
        self.CoordinatesPredictorY = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.ConvTranspose1d(256, 128, 3, 3, 0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.ConvTranspose1d(128, 64, 3, 3, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.ConvTranspose1d(64, 32, 3, 3, 0, bias=False),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),

            nn.ConvTranspose1d(32, 16, 3, 3, 0, bias=False),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),

            nn.ConvTranspose1d(16, 1, 9, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        

    def forward(self, z, energy):
        seed = self.TransConv(z*energy)
        coordinates = self.CoordinatesPredictor(seed)
        energy = self.EnergyPredictor(seed)
        coordinatesY = self.CoordinatesPredictorY(seed)
        
        event = torch.cat((coordinates[:, 0:1], coordinatesY, coordinates[:, 1:2], energy), dim=1)
        
        return event




class GeneratorS(nn.Module):
    def __init__(self, cfg):
        super(GeneratorS, self).__init__()

        self.TransConv = nn.Sequential(

            nn.ConvTranspose1d(100, 8192, 2, 1, 0, bias=False),
            # nn.BatchNorm1d(8192),
            # nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(8192, 4096, 3, 3, 0, bias=False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(4096, 2048, 3, 3, 1, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(2048, 1024, 3, 3, 0, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(True)
        )
        
        self.CoordinatesPredictor = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.ConvTranspose1d(256, 2, 3, 3, 0, bias=False)
        )
        
        self.EnergyPredictor = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(True),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 1, 3, 3, 0, bias=False),
            nn.ReLU()
        )
        
        self.CoordinatesPredictorY = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.ConvTranspose1d(256, 1, 3, 3, 0, bias=False)
        )
        

    def forward(self, z, energy):
        seed = self.TransConv(z*energy)
        coordinates = self.CoordinatesPredictor(seed)
        energy = self.EnergyPredictor(seed)
        coordinatesY = self.CoordinatesPredictorY(seed)
        
        event = torch.cat((coordinates[:, 0:1], coordinatesY, coordinates[:, 1:2], energy), dim=1)
        
        return event