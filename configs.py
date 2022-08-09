class Configs():
    
    def __init__(self):
        
    # Experiment Name
        self.name = '4channels_seeded_GAN_34992_len'

    # Generator
        self.nz = 100
        self.noise = 'normal'
        # labels for training
        self.Emax = 1
        self.Emin = 1

    # Discriminator
        self.num_discriminators = 10
        self.nfd = 32

    # Dtataloader
        self.dataPath = '/beegfs/desy/user/akorol/projects/getting_high/ILDConfig/StandardConfig/production/out/no_projection.hdf5'
        self.workers = 128
        self.bs = 32
        self.pin_memory = True
        self.shuffle = True

    # Optimizer
        self.n_epochs = 200
        self.lrG = 0.000005
        self.lrD = 0.000001
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.weight_decayD = 0
        self.weight_decayG = 0
        self.generator_loss = 'BCE'
        self.gLossFactor = 1
        self.gamma = 25 


    # Others
        self.device = 'cuda'
        # self.checkpoint = '/beegfs/desy/user/akorol/trained_models/4channels_seeded_sGAN_WU-False_bs32_lr0.0001Gx1_L2RD0.1_L2RG0_LossBCE_gamma25_uniformNoise_fake2realRatio1/epoch_200.pth'
        self.checkpoint = None
        self.wormup = False
        self.path2save = '/beegfs/desy/user/akorol/trained_models'
        self.seed = 42

        