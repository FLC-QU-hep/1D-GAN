class Configs():
    
    def __init__(self):
        
    # Experiment Name
        self.name = '4channels_seeded_WGANGP'

    # Generator
        self.nz = 100
        self.noise = 'normal'
        # labels for training
        self.Emax = 1
        self.Emin = 1
        self.wormup = False
        self.wormup_bs = 1024
        self.wormup_iterations = 3000
        self.wormup_lr = 0.0001

    # Discriminator
        self.nfd = 32
        self.critic_iterations = 10
        self.GP_WEIGHT = 10

    # Dtataloader
        self.dataPath = '/beegfs/desy/user/akorol/projects/getting_high/ILDConfig/StandardConfig/production/out/no_projection.hdf5'
        self.workers = 128
        self.bs = 128
        self.pin_memory = True
        self.shuffle = True

    # Optimizer
        self.n_epochs = 200
        self.lrG = 0.0000005
        self.lrD = 0.0000005
        self.beta1 = 0
        self.beta2 = 0.999
        self.weight_decayD = 0
        self.weight_decayG = 0


    # Others
        self.device = 'cuda'
        # self.checkpoint = '/beegfs/desy/user/akorol/trained_models/4channels_seeded_sGAN_WU-False_bs32_lr0.0001Gx1_L2RD0.1_L2RG0_LossBCE_gamma25_uniformNoise_fake2realRatio1/epoch_200.pth'
        self.checkpoint = None
        self.path2save = '/beegfs/desy/user/akorol/trained_models'
        self.seed = 42

        