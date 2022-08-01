
import h5py
from torch.utils.data import Dataset
import torch

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

        
        

    def __getitem__(self, idx):
        
        event = self.dataset['events'][idx][:, -1296:]
        
        event[2, :][event[2, :] > 0] = event[2, :][event[2, :] > 0] - 40 # z shift
        event[3, :] = event[3, :] * 1000 # energy scale
        
        event[0, :] = (event[0, :] - self.Xmin) / (self.Xmax - self.Xmin) # x
        event[1, :] = (event[1, :] - self.Ymin) / (self.Ymax - self.Ymin) # y
        event[2, :] = (event[2, :] - self.Zmin) / (self.Zmax - self.Zmin) # z

        event = event[[0, 1, 2, 3]]
        
        if self.chech_overfiting:
            return {'event' : event,
                    'z' : self.z[idx],
                    'energy' : 1}
        else:
            return {'event' : event,
                    'energy' : 1}

    def __len__(self):
        return len(self.dataset['events'])

