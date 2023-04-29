import torch
from torch import nn

class MelGanDiscriminatorLosses:

    @staticmethod
    def feature_loss(fmap_r, fmap_g) :
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2
    
    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses
    
    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

class MelGanDiscriminatorLosses:

    @staticmethod
    def generator_loss(real_pred, fake_pred):
        mse = nn.MSELoss()
        gen_loss = 0
        for i in range(len(fake_pred)) :
            gen_loss += mse(torch.ones_like(fake_pred[i][-1]), fake_pred[i][-1])

        return gen_loss / 3

    @staticmethod
    def feature_matching_loss(real_pred, fake_pred):
        mae = nn.L1Loss()
        fm_loss = 0
        for i in range(len(fake_pred)):
            for j in range(len(fake_pred[i]) - 1):
                fm_loss += mae(real_pred[i][j], fake_pred[i][j])
                fm_loss = fm_loss / (len(fake_pred[i]) - 1)
        return fm_loss / (len(fake_pred))

    @staticmethod
    def discriminator_loss(real_pred, fake_pred):
        mse = nn.MSELoss()
        real_loss = 0
        fake_loss = 0
        for i in range(len(real_pred)):
            real_loss += mse(torch.ones_like(real_pred[i][-1]), real_pred[i][-1])
            fake_loss += mse(torch.ones_like(fake_pred[i][-1]), fake_pred[i][-1])

        disc_loss = (real_loss + fake_loss) / len(real_pred)
        return disc_loss