import torch
import torch.nn as nn
from models import GNet, DNet
from dataset import HDF5Dataset
from configs_WGAN import Configs
from models.utils import weights_init, get_gradient_penalty

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


def save(netG, netD, omtim_G, optim_D, epoch, loss, path_to_save):
    torch.save({
                'Generator': netG.state_dict(),
                'Discriminator': netD.state_dict(),
                'G_optimizer': omtim_G.state_dict(),
                'D_optimizer': optim_D.state_dict(),
                'epoch': epoch,
                'loss': loss
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
train_folder = f'{cfg.path2save}/{cfg.name}/WU-{cfg.wormup}_bs{cfg.bs}_lrD{cfg.lrD}G{cfg.lrG}_L2RD{cfg.weight_decayD}_L2RG{cfg.weight_decayG}_{cfg.noise}Noise'
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
print(f'train_folder: {train_folder}')

with open(f'{train_folder}/log.txt', "w") as file:
        file.write(f'[CONFIGS] workers: {cfg.workers}, bs: {cfg.bs}, beta1: {cfg.beta1}, lrD: {cfg.lrD}, lrG:  {cfg.lrG}, pin_memory: {cfg.pin_memory}\n')
        file.write(f'checkpoint: {cfg.checkpoint}\n')



data = HDF5Dataset.PointCloudDataset(cfg.dataPath)
dataloader = torch.utils.data.DataLoader(data, batch_size=cfg.bs, pin_memory = cfg.pin_memory,
                                         shuffle=cfg.shuffle, num_workers=cfg.workers, generator=g)



netG = GNet.Generator(cfg)
netG.apply(weights_init)
netG.to(device)
# netG = nn.DataParallel(netG)

if cfg.wormup:
    optimizer_G_wp = torch.optim.Adam(netG.parameters(), lr=cfg.wormup_lr, betas=(cfg.beta1, cfg.beta2))
    mse = nn.MSELoss()

    dataloader_wp = torch.utils.data.DataLoader(data, batch_size=cfg.wormup_bs, shuffle=cfg.shuffle)
    batch = next(iter(dataloader_wp))

    real_showers = batch['event'].float().to(device)

    for i in range(cfg.wormup_iterations):

        z, e_labels = get_gen_seed(cfg.wormup_bs)
        fake_shower = netG(z, e_labels)

        loss = mse(real_showers, fake_shower)
        loss.backward()
        optimizer_G_wp.step()

    
netD = DNet.Discriminator(cfg)
netD.apply(weights_init)
netD.to(device)
# netD = nn.DataParallel(netD)



optimizer_G = torch.optim.Adam(netG.parameters(), lr=cfg.lrG, betas=(cfg.beta1, cfg.beta2))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=cfg.lrD, betas=(cfg.beta1, cfg.beta2))


G_losses = np.array([])
D_GP = np.array([])
D_losses = np.array([])

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
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write('[INFO] Done\n')
    else:
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write('[INFO] Loading from checkpoint...\n')
        checkpoint = torch.load(cfg.checkpoint)
        netG.load_state_dict(checkpoint['Generator'])
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write('[INFO] Done\n')

def train(real_showers, netG, netD):
    
    bs = len(real_showers)
    
    ######################################################
    # Train Critic
    ######################################################

    mean_iteration_critic_loss = 0
    mean_GP = 0 
    for _ in range(cfg.critic_iterations):
        netD.zero_grad()

        z, e_labels = get_gen_seed(bs)
        fake_shower = netG(z, e_labels)

        critic_pred_real = netD(real_showers).view(-1)
        critic_pred_fake = netD(fake_shower).view(-1)

        epsilon = torch.rand(len(fake_shower), 1, 1, device=device, requires_grad=True)
        gp = get_gradient_penalty(netD, real_showers, fake_shower.detach(), epsilon)

        loss_critic = torch.mean(critic_pred_fake) - torch.mean(critic_pred_real) + gp*cfg.GP_WEIGHT
        loss_critic.backward(retain_graph=True)

        optimizer_D.step()

        mean_iteration_critic_loss += loss_critic.item() / cfg.critic_iterations
        mean_GP += (gp*cfg.GP_WEIGHT).item() / cfg.critic_iterations

    ######################################################
    # Train Generator
    ######################################################
    netG.zero_grad()

    z, e_labels = get_gen_seed(bs)
    fake_shower = netG(z, e_labels)

    critic_pred_fake = netD(fake_shower).reshape(-1)
    loss_gen = -torch.mean(critic_pred_fake)
    loss_gen.backward()

    optimizer_G.step()


    return loss_gen.item(), mean_iteration_critic_loss, mean_GP




if __name__ == '__main__':

    for epoch in range(cfg.n_epochs):
        epoch += eph + 1
        time_start = time()
        for i, batch in enumerate(dataloader):
            real_showers = batch['event'].float().to(device)
    #         real_energys = batch['energy'].float().to(device)

            lossG, lossD, mean_GP = train(real_showers, netG, netD)

            # Output training stats
            G_losses = np.append(G_losses, lossG)
            D_losses = np.append(D_losses, lossD)
            D_GP = np.append(D_GP, mean_GP)
            

        t = round(time() - time_start)
        t_per_sample = round(t/len(dataloader)/cfg.bs, 5)
        print(f'Epoch {epoch} finished, time: {t} s / epoch; {t_per_sample} s / sample')
        print(f'[{epoch}/{cfg.n_epochs}], (Loss_D: {lossD})  (Loss_G: {lossG}) \n\n\n')
        with open(f'{train_folder}/log.txt', "a") as file:
            file.write(f'[INFO] time: {t} s / epoch; {t_per_sample} s / sample\n')
            file.write(f'[{epoch}/{cfg.n_epochs}], (Loss_D: {lossD})  (Loss_G: {lossG}) \n\n')


        if epoch%5 == 0:
            PATH_save = f'{train_folder}/epoch_{epoch}.pth'

            # loss =  np.array([G_losses, D_losses])
            loss = {'G_losses': G_losses, 'D_losses': D_losses, 'D_GP': D_GP}

            save(netG=netG, netD=netD, omtim_G=optimizer_G, optim_D=optimizer_D, epoch=epoch, loss=loss,
                path_to_save=PATH_save)