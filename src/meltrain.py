import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torch import nn

from src.data.dataset import AudioDataset
import auraloss

from src.models.model import SwarModel
from src.models.discriminators import MelMultiScaleDiscriminator
from src.utils.loss import *

dataset = AudioDataset("E:/Programming/major/200/*.wav", "")
stft_loss = auraloss.freq.STFTLoss()

def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    data = nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return data.transpose(1, 2)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn, num_workers=0)

model = SwarModel.build_model(6).to(torch.device('cuda'))
disc = MelMultiScaleDiscriminator().to(torch.device('cuda'))
generator_optim = optim.Adam(model.parameters(), lr=3 * 1e-4, betas=(0.5, 0.9))
discriminator_optim = optim.Adam(disc.parameters(), lr=1e-6, betas=(0.5, 0.9))

path = "./models/encodec-gan-v-8.pt"

checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
disc.load_state_dict(checkpoint['disc_state_dict'])
generator_optim.load_state_dict(checkpoint['genoptimizer_state_dict'])
discriminator_optim.load_state_dict(checkpoint['discoptimizer_state_dict'])

loss = MelGanDiscriminatorLosses()

next_path = "encodec-gan-v-9.pt"

def save_model(model, disc, gen_optimizer, disc_optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'genoptimizer_state_dict': gen_optimizer.state_dict(),
        'discoptimizer_state_dict': disc_optimizer.state_dict()
    },next_path)

for epoch in range(0, 2000):
  print("EPOCH number = %d" % epoch)
  for i, x in enumerate(iter(dataloader)):
    with torch.autograd.set_detect_anomaly(True):

      x = x.to(torch.device("cuda"))

      fake_audio, vq_loss = model(x)

      # Generator
      disc_fake = disc(fake_audio)
      disc_real = disc(x)

      gen_loss1 = stft_loss(x, fake_audio) + vq_loss
      gen_loss2 = loss.generator_loss(disc_real, disc_fake)
      gen_loss3 = loss.feature_matching_loss(disc_real, disc_fake)

      gen_loss = gen_loss1 + gen_loss2 + 100 * gen_loss3

      generator_optim.zero_grad()
      gen_loss.backward()
      generator_optim.step()

      # Discriminator
      disc_fake = disc(fake_audio.detach())
      disc_real = disc(x)
      
      disc_loss = loss.discriminator_loss(disc_real, disc_fake)

      discriminator_optim.zero_grad()
      disc_loss.backward()
      discriminator_optim.step()

      if i % 5 == 0:
        print("Training generator loss = ", gen_loss) 

    save_model(model, disc, generator_optim, discriminator_optim)