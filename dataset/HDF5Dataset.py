
import h5py
from torch.utils.data import Dataset
import torch
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, file_path, chech_overfiting=False):
        self.dataset = h5py.File(file_path, 'r')
        self.Ymin = 0
        self.Ymax = 0.5250244140625*30
#         self.Ymax = 2010.5699-1811
        self.Xmin = -200
        self.Xmax = 200
        self.Zmin = -160 -40
        self.Zmax = 240 -40
        self.chech_overfiting = chech_overfiting

        if chech_overfiting:
            self.z = torch.FloatTensor(len(self.dataset['events']), 100, 1).uniform_(-1, 1)
            
        self.cell_thickness = 0.5250244140625
        self.full_length = self.cell_thickness*30
        self.first_center = self.cell_thickness/self.full_length/2
        self.gap = 0.0001
        

    def __getitem__(self, idx):
        
        # event = self.dataset['events'][idx][:, -34992:]
        # event = self.dataset['events'][idx][:, :453]
        event = self.dataset['events'][idx][:, :2373]
        
        event[2, :][event[2, :] > 0] = event[2, :][event[2, :] > 0] - 40 # z shift to the origin
        event[3, :] = event[3, :] * 1000 # energy scale
        
        event[0, :] = (event[0, :] - self.Xmin) / (self.Xmax - self.Xmin) # x coordinate normalization
        event[1, :] = (event[1, :] - self.Ymin) / (self.Ymax - self.Ymin) # y coordinate normalization
        event[2, :] = (event[2, :] - self.Zmin) / (self.Zmax - self.Zmin) # z coordinate normalization
        

        ######################################################
        # random distribution of the y coordinate inside cell 
        ######################################################
        y = event[1, :]
        for i in range(30):
            layer_center = self.cell_thickness*i/self.full_length + self.first_center
            layer_start = self.cell_thickness*i/self.full_length
            n = len(y[(y < layer_center + self.gap) & (y > layer_center - self.gap)])
            if n > 0:
                new_y = np.random.uniform(layer_start, layer_start+self.cell_thickness/self.full_length, n)
                y[(y < layer_center + self.gap) & (y > layer_center - self.gap)] = new_y
                event[1, :] = y
        ######################################################
        ######################################################
        
        event = event[[0, 1, 2, 3]]
        
        if self.chech_overfiting:
            return {'event' : event,
                    'z' : self.z[idx],
                    'energy' : self.dataset['energy'][idx]}
        else:
            return {'event' : event,
                    'energy' : self.dataset['energy'][idx]}

    def __len__(self):
        return len(self.dataset['events'])

