class Configs():
    
    def __init__(self):
        
    # Experiment Name
        self.name = '4channels_seeded_sGAN'

    # Generator
        self.nz = 100
        self.noise = 'uniform'
        # labels for training
        self.Emax = 1
        self.Emin = 1
    # Discriminator
        self.nfd = 32

    # Dtataloader
        self.dataPath = '/beegfs/desy/user/akorol/projects/getting_high/ILDConfig/StandardConfig/production/out/no_projection.hdf5'
        self.workers = 64
        self.bs = 32
        self.pin_memory = True
        self.shuffle = True
        self.fake2realRatio = 3     

    # Optimizer
        self.n_epochs = 200
        self.lr = 0.0001
        self.beta1 = 0.5
        self.generator_loss = 'MSE+BCE'
        self.gLossFactor = 1
        self.gamma = 0.3


    # Others
        self.device = 'cuda'
        # self.checkpoint = '/beegfs/desy/user/akorol/trained_models/overfitting_bs1024_lr0.0001Gx1_LossMSE/epoch_20000.pth'
        self.checkpoint = None
        self.wormup = False
        self.path2save = '/beegfs/desy/user/akorol/trained_models'
        self.seed = 42

        