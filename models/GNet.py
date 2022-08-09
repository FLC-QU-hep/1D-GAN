import torch
import torch.nn as nn


class SIN(nn.Module):
    def __init__(self):
        super().__init__()

    def sin(self, input):
        return torch.sin(input)

    def forward(self, input):
        return self.sin(input) 



class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()

        self.TransConv = nn.Sequential(

            nn.ConvTranspose1d(cfg.nz, 8192, 2, 1, 0, bias=False),
            nn.BatchNorm1d(8192),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose1d(8192, 4096, 3, 3, 0, bias=False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose1d(4096, 2048, 3, 3, 1, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2, True),
            
            nn.ConvTranspose1d(2048, 1024, 3, 3, 0, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True)
        )
        
        self.CoordinatesGeneratorX = nn.Sequential(
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

            nn.ConvTranspose1d(32, 1, 3, 3, 0, bias=False),
            nn.Sigmoid()

        )
        
        self.EnergyGenerator = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 3, 3, 0, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 3, 3, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 32, 3, 3, 0, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),

            nn.ConvTranspose1d(32, 1, 3, 3, 0, bias=False),
            nn.ReLU()

        )
        
        self.CoordinatesGeneratorY = nn.Sequential(
            nn.ConvTranspose1d(1024, 512, 3, 3, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(512, 256, 3, 3, 0, bias=False),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),

            nn.ConvTranspose1d(256, 128, 3, 3, 0, bias=False),
            nn.BatchNorm1d(128),
            nn.Sigmoid(),
            
            nn.ConvTranspose1d(128, 64, 3, 3, 0, bias=False),
            nn.BatchNorm1d(64),
            nn.Sigmoid(),

            nn.ConvTranspose1d(64, 32, 3, 3, 0, bias=False),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),

            nn.ConvTranspose1d(32, 1, 3, 3, 0, bias=False),
            nn.Sigmoid()
        )
        
        self.CoordinatesGeneratorZ = nn.Sequential(
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

            nn.ConvTranspose1d(32, 1, 3, 3, 0, bias=False),
            nn.Sigmoid()

        )
        

    def forward(self, z, energy):
        seed = self.TransConv(z*energy)
        energy = self.EnergyGenerator(seed)
        coordinatesX = self.CoordinatesGeneratorX(seed)
        coordinatesY = self.CoordinatesGeneratorY(seed)
        coordinatesZ = self.CoordinatesGeneratorZ(seed)
        
        event = torch.cat((coordinatesX, coordinatesY, coordinatesZ, energy), dim=1)
        
        return event



class GeneratorS(nn.Module):
    def __init__(self, cfg):
        super(GeneratorS, self).__init__()

        self.TransConv = nn.Sequential(

            nn.ConvTranspose1d(cfg.nz, 8192, 2, 1, 0, bias=False),
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

            nn.ConvTranspose1d(256, 2, 3, 3, 0, bias=False),
            nn.Sigmoid(),
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

            nn.ConvTranspose1d(256, 1, 3, 3, 0, bias=False),
            nn.Sigmoid()
        )
        

    def forward(self, z, energy):
        seed = self.TransConv(z*energy)
        coordinates = self.CoordinatesPredictor(seed)
        energy = self.EnergyPredictor(seed)
        coordinatesY = self.CoordinatesPredictorY(seed)
        
        event = torch.cat((coordinates[:, 0:1], coordinatesY, coordinates[:, 1:2], energy), dim=1)
        
        return event
