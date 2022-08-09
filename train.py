import torch
import torch.nn as nn
from models import GNet, DNet
from dataset import HDF5Dataset
from configs import Configs
from models.utils import weights_init

# import matplotlib.pyplot as plt
import random
import numpy as np

from time import time
import os

cfg = Configs()
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
g = torch.Generator()
g.manual_seed(cfg.seed)
device = torch.device(cfg.device)


def save(netG, netD, omtim_G, optim_D, epoch, loss, scores, path_to_save):
    torch.save({
                'Generator': netG.state_dict(),
                'Discriminator': netD.state_dict(),
                'G_optimizer': omtim_G.state_dict(),
                'D_optimizer': optim_D.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'D_scores': scores,
                # 'z' : z
                },
                path_to_save)

def get_gen_seed(bs, noise=cfg.noise):
    if noise == 'uniform':
        z = torch.FloatTensor(int(bs), cfg.nz, 1).uniform_(-1, 1).to(device)
    else:
        z = torch.FloatTensor(int(bs), cfg.nz, 1).normal_(0, 1).to(device)
    e_labels = torch.FloatTensor(bs, 1, 1).uniform_(cfg.Emax, cfg.Emin).to(device)

    return z, e_labels


print(f'checkpoint: {cfg.checkpoint}')
train_folder = f'{cfg.path2save}/{cfg.name}/WU-{cfg.wormup}_bs{cfg.bs}_lr{cfg.lrD}G{cfg.lrG}_L2RD{cfg.weight_decayD}_L2RG{cfg.weight_decayG}_Loss{cfg.generator_loss}_gamma{cfg.gamma}_{cfg.noise}Noise_num_discriminators{cfg.num_discriminators}'
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
print(f'train_folder: {train_folder}')

with open(f'{train_folder}/log.txt', "w") as file:
        file.write(f'[CONFIGS] workers: {cfg.workers}, bs: {cfg.bs}, beta1: {cfg.beta1}, lr{cfg.lrD}G{cfg.lrG}, pin_memory: {cfg.pin_memory}, netG loss: {cfg.generator_loss}, gLossFactor: {cfg.gLossFactor}\n')
        file.write(f'checkpoint: {cfg.checkpoint}\n')




data = HDF5Dataset.PointCloudDataset(cfg.dataPath)
dataloader = torch.utils.data.DataLoader(data, batch_size=cfg.bs, pin_memory = cfg.pin_memory,
                                         shuffle=cfg.shuffle, num_workers=cfg.workers, generator=g)




netG = GNet.Generator(cfg)
netG.apply(weights_init)
netG.to(device)
    
D_nets = []
D_optimizers = []
for i in range(cfg.num_discriminators):
    netD = DNet.Discriminator(cfg)
    netD.apply(weights_init)
    netD.to(device)
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=cfg.lrD, betas=(cfg.beta1, 0.999), weight_decay=cfg.weight_decayD)
    D_nets.append(netD)
    D_optimizers.append(optimizer_D)


optimizer_G = torch.optim.Adam(netG.parameters(), lr=cfg.lrG, betas=(cfg.beta1, 0.999), weight_decay=cfg.weight_decayG)

criterion = nn.BCEWithLogitsLoss()
MSEcriterion = nn.MSELoss()

G_losses = np.array([])
G_loss_BCE = np.array([])
G_loss_MSE = np.array([])
D_losses = np.array([])
D_scores_x = np.array([])
D_scores_z1 = np.array([])
D_scores_z2 = np.array([])

eph = 0


if cfg.checkpoint:
    if not cfg.wormup:
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write('[INFO] Loading from checkpoint...\n')
        checkpoint = torch.load(cfg.checkpoint)
        netG.load_state_dict(checkpoint['Generator'])
        netD.load_state_dict(checkpoint['Discriminator'])
        optimizer_G.load_state_dict(checkpoint['G_optimizer'])
        optimizer_D.load_state_dict(checkpoint['D_optimizer'])
        eph = checkpoint['epoch']
        chechoint_loss = checkpoint['loss']
        G_losses = chechoint_loss['G_losses']
        D_losses = chechoint_loss['D_losses']
        chechoint_scores = checkpoint['D_scores']
        D_scores_x = chechoint_scores[0]
        D_scores_z1 = chechoint_scores[1]
        D_scores_z2 = chechoint_scores[2]
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write('[INFO] Done\n')
    else:
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write('[INFO] Loading from checkpoint...\n')
        checkpoint = torch.load(cfg.checkpoint)
        netG.load_state_dict(checkpoint['Generator'])
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write('[INFO] Done\n')



def train(real_showers, netG, D_nets, D_optimizers, epoch, noise=cfg.noise):
    bs = len(real_showers)

    # Adversarial ground truths
    valid_label = torch.FloatTensor(np.ones(bs)).to(device)
    fake_label = torch.FloatTensor(np.zeros(int(bs))).to(device)

    ######################################################
    # Train Discriminator
    ######################################################
    errD_mean = 0
    for netD, optimizer_D in list(zip(D_nets, D_optimizers)):
        netD.zero_grad()

        output = netD(real_showers).view(-1)
        errD_real = criterion(output, valid_label)
        errD_real.backward()
        D_x = output.mean().item()

        z, e_labels = get_gen_seed(bs)
        fake_shower = netG(z, e_labels)

        output = netD(fake_shower).view(-1)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()
        errD = (errD_real.item() + errD_fake.item())/2
        
        errD_mean += errD / len(D_nets)

        optimizer_D.step()

    if cfg.wormup:
        if epoch == 1:
            D_G_z2 = D_G_z1

            errG = errD_real

            return errD_mean, D_x, D_G_z1, D_G_z2, errG

    ######################################################
    # Train Generator
    ######################################################
    netG.zero_grad()

    if cfg.generator_loss == "MSE":

        embeding_fake = netD.MainConv(fake_shower)
        embeding_real = netD.MainConv(real_showers)

        errG_total = MSEcriterion(embeding_fake, embeding_real)
        
        D_G_z2 = 0

        errG_total.backward(retain_graph=True)
        optimizer_G.step()
    
    else:
        errG_total = 0
        for netD in (D_nets):
            z, e_labels = get_gen_seed(bs)
            fake_shower = netG(z, e_labels)
            output = netD(fake_shower).view(-1)
            valid_label = torch.FloatTensor(np.ones(len(output))).to(device)
            errG_total += criterion(output, valid_label)
        errG_total.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        optimizer_G.step()

    return errD_mean, D_x, D_G_z1, D_G_z2, errG_total.item()




if __name__ == '__main__':

    for epoch in range(cfg.n_epochs):
        epoch += eph + 1
        time_start = time()
        for i, batch in enumerate(dataloader):
            real_showers = batch['event'].float().to(device)
    #         real_energys = batch['energy'].float().to(device)

            errD, D_x, D_G_z1, D_G_z2, errG = train(real_showers, netG, D_nets, D_optimizers, epoch)

            # Output training stats
            G_losses = np.append(G_losses, errG)
            D_losses = np.append(D_losses, errD)
            D_scores_x = np.append(D_scores_x, D_x)
            D_scores_z1 = np.append(D_scores_z1, D_G_z1)
            D_scores_z2 = np.append(D_scores_z2, D_G_z2)


        t = round(time() - time_start)
        t_per_sample = round(t/len(dataloader)/cfg.bs, 5)
        print(f'Epoch {epoch} finished, time: {t} s / epoch; {t_per_sample} s / sample')
        print('[%d/%d], (Loss_D: %.4f)  (Loss_G: %.4f),  (D(x): %.4f)  (D(G(z)): %.4f / %.4f)  \n\n\n'
                    % (epoch, cfg.n_epochs, errD, errG, D_x, D_G_z1, D_G_z2))
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write(f'[INFO] time: {t} s / epoch; {t_per_sample} s / sample\n')
            file.write('[INFO] Epoch [%d/%d], (Loss_D: %.4f)  (Loss_G: %.4f),  (D(x): %.4f)  (D(G(z)): %.4f / %.4f)  \n\n'
                    % (epoch, cfg.n_epochs, errD, errG, D_x, D_G_z1, D_G_z2))


        if epoch%5 == 0:
            PATH_save = f'{train_folder}/epoch_{epoch}.pth'

            # loss =  np.array([G_losses, D_losses])
            loss = {'G_losses': G_losses, 'D_losses': D_losses}
            D_scores = np.array([D_scores_x, D_scores_z1, D_scores_z2])

            save(netG=netG, netD=netD, omtim_G=optimizer_G, optim_D=optimizer_D, epoch=epoch, loss=loss, scores=D_scores,
                path_to_save=PATH_save)